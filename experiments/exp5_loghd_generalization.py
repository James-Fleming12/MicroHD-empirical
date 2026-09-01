"""Exp5a: stress-test LogHD (class-axis compression) on the generalization tasks.

LogHD (Yun et al. 2026) replaces the C class prototypes with n ~= ceil(log_k C)
bundle hypervectors and decodes in an n-dimensional activation space, cutting
classifier memory to O(D log C) *while preserving dimensionality D*. Unlike the
feature-axis compressors of exp2/exp3, the encoder is left untouched, so the
encoder-side geometry (novel-class structure, pairwise geometry fidelity) should
be preserved. But the classifier-side (ID accuracy, domain-shift OOD robustness,
OOD detection) now rides on the n-coordinate activation space built by
superposing ~C/2 prototypes into each bundle.

Questions:
  * Does class-axis compression preserve what feature-axis compression destroys?
  * Where is the limit? (class count C, class separation, redundancy n - ceil log)
  * At matched classifier memory, class-axis (LogHD) vs feature-axis (pruning)?

Also sweeps C and class separation to map the activation-space SNR regime, and
checks the paper's optional bundle refinement.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.data import make_domain_shift_task, make_novel_class_task
from modules.dpqhd import CompressedHDCModel, TruncatedEncoder, truncate_class_hvs
from modules.hdc import HDClassifier, ProjectionEncoder
from modules.loghd import (
    LogHDClassifier,
    activation_margin,
    loghd_memory_bits,
    matched_feature_keep,
)
from modules.metrics import (
    accuracy,
    clustering_metrics,
    geometry_fidelity,
    ood_detection_auroc,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASE_DIM = 10000
REDUNDANCY = (0, 1, 2)  # extra bundles beyond ceil(log_k C)

def base_task(name: str, seed: int):
    return make_domain_shift_task(seed=seed) if name == "shift" else make_novel_class_task(seed=seed)

def train_splits(task, name: str):
    if name == "shift":
        return task.x_train, task.y_train, int(task.y_train.max()) + 1
    x = torch.cat([task.x_train, task.x_val])
    y = torch.cat([task.y_train, task.y_val])
    return x, y, task.num_known_classes

def cosine_margin(enc, class_hvs, x, y) -> float:
    h = enc.encode(x).float()
    cn = F.normalize(class_hvs.to(h.device), dim=1)
    s = h @ cn.T
    n = y.shape[0]
    idx = torch.arange(n, device=s.device)
    sc = s[idx, y]
    so = s.clone()
    so[idx, y] = -float("inf")
    return float((sc - so.max(dim=1).values).mean())

def eval_shift(task, model, *, loghd: bool) -> dict:
    """Common evaluation on the domain-shift task."""
    row = {"val_acc": accuracy(model.predict(task.x_val), task.y_val),
           "id_acc": accuracy(model.predict(task.x_test_src), task.y_test_src)}
    tgt = []
    for name, xt, yt in task.targets:
        a = accuracy(model.predict(xt), yt)
        row[f"acc_{name}"] = a
        tgt.append(a)
    row["avg_ood_acc"] = sum(tgt) / len(tgt)
    if loghd:
        row["margin_id"] = activation_margin(model, task.x_test_src, task.y_test_src)
        row["margin_ood"] = sum(
            activation_margin(model, xt, yt) for _, xt, yt in task.targets) / len(task.targets)
    return row

def eval_novel(task, model, *, loghd: bool) -> dict:
    row = {"val_acc": accuracy(model.predict(task.x_val), task.y_val),
           "id_acc": accuracy(model.predict(task.x_test_id), task.y_test_id)}
    h_nov = model.encoder.encode(task.x_novel)
    cl = clustering_metrics(h_nov, task.num_novel_classes, task.y_novel.numpy(), seed=0)
    row.update(nmi=cl["nmi"], hungarian_acc=cl["hungarian_acc"])
    if loghd:
        s_id = model.ood_score(task.x_test_id).cpu().numpy()
        s_no = model.ood_score(task.x_novel).cpu().numpy()
        row["auroc"] = ood_detection_auroc(s_id, s_no)
    else:
        s_id = model.scores(task.x_test_id).max(dim=1).values.cpu().numpy()
        s_no = model.scores(task.x_novel).max(dim=1).values.cpu().numpy()
        row["auroc"] = ood_detection_auroc(s_id, s_no)
    return row

def run_grid(task_name: str, seed: int, epochs: int) -> list[dict]:
    task = base_task(task_name, seed)
    x_tr, y_tr, n_cls = train_splits(task, task_name)
    enc = ProjectionEncoder(x_tr.shape[1], BASE_DIM, seed=seed)
    enc.fit_feature_range(x_tr)
    clf = HDClassifier(enc, n_cls).fit(x_tr, y_tr, epochs=epochs, seed=seed)
    w_full = clf.class_hvs.clone()

    rows = []
    geom = geometry_fidelity(enc, task.x_val[:1000])

    base_eval = eval_shift(task, clf, loghd=False) if task_name == "shift" \
        else eval_novel(task, clf, loghd=False)
    rows.append(dict(model="baseline", k="", n="", eps="", keep_frac=1.0,
                     mem_kb=round(n_cls * BASE_DIM * 32 / 8 / 1024, 1),
                     mem_frac=1.0, geom_fid=geom, **base_eval))

    for k in (2, 3):
        n_min = math.ceil(math.log(n_cls) / math.log(k))
        for eps in REDUNDANCY:
            n = n_min + eps
            lc = LogHDClassifier(enc, n_cls, alphabet=k, num_bundles=n).fit(
                x_tr, y_tr, seed=seed)
            ev = eval_shift(task, lc, loghd=True) if task_name == "shift" \
                else eval_novel(task, lc, loghd=True)
            rows.append(dict(model="loghd", k=k, n=n, eps=eps, keep_frac="",
                             mem_kb=round(loghd_memory_bits(n_cls, BASE_DIM, n) / 8 / 1024, 1),
                             mem_frac=round(loghd_memory_bits(n_cls, BASE_DIM, n) / (n_cls * BASE_DIM * 32), 4),
                             geom_fid=geom, **ev))

            keep = matched_feature_keep(n_cls, BASE_DIM, n)
            enc_t = TruncatedEncoder(enc, keep)
            w_t = truncate_class_hvs(w_full, keep)
            fm = CompressedHDCModel(enc_t, w_t)
            ev = eval_shift(task, fm, loghd=False) if task_name == "shift" \
                else eval_novel(task, fm, loghd=False)
            rows.append(dict(model="feataxis", k=k, n=n, eps=eps, keep_frac=round(keep / BASE_DIM, 4),
                             mem_kb=round(n_cls * keep * 32 / 8 / 1024, 1),
                             mem_frac=round(n_cls * keep * 32 / (n_cls * BASE_DIM * 32), 4),
                             geom_fid=geometry_fidelity(enc_t, task.x_val[:1000]), **ev))

    for row in rows:
        row.update(task=task_name, seed=seed)
    return rows


def run_class_sweep(seed: int, epochs: int) -> list[dict]:
    """Class-count scaling: the activation space grows only logarithmically in C."""
    rows = []
    for C in (5, 12, 20, 30):
        task = make_domain_shift_task(seed=seed, num_classes=C)
        x_tr, y_tr = task.x_train, task.y_train
        enc = ProjectionEncoder(x_tr.shape[1], BASE_DIM, seed=seed)
        enc.fit_feature_range(x_tr)
        clf = HDClassifier(enc, C).fit(x_tr, y_tr, epochs=epochs, seed=seed)
        ood_b = sum(accuracy(clf.predict(xt), yt) for _, xt, yt in task.targets) / 3
        rows.append(dict(seed=seed, C=C, model="baseline", k="", n="",
                         val_acc=accuracy(clf.predict(task.x_val), task.y_val),
                         avg_ood_acc=ood_b))
        for k in (2, 3):
            n = math.ceil(math.log(C) / math.log(k))
            lc = LogHDClassifier(enc, C, alphabet=k, num_bundles=n).fit(x_tr, y_tr, seed=seed)
            ood = sum(accuracy(lc.predict(xt), yt) for _, xt, yt in task.targets) / 3
            rows.append(dict(seed=seed, C=C, model="loghd", k=k, n=n,
                             val_acc=accuracy(lc.predict(task.x_val), task.y_val),
                             avg_ood_acc=ood))
    return rows


def run_spread_sweep(seed: int, epochs: int) -> list[dict]:
    """Class-separation scaling: activation-space SNR is set by separation."""
    rows = []
    for spread in (0.7, 1.0, 1.5, 2.5):
        task = make_domain_shift_task(seed=seed, num_classes=30, spread=spread)
        x_tr, y_tr = task.x_train, task.y_train
        enc = ProjectionEncoder(x_tr.shape[1], BASE_DIM, seed=seed)
        enc.fit_feature_range(x_tr)
        clf = HDClassifier(enc, 30).fit(x_tr, y_tr, epochs=epochs, seed=seed)
        rows.append(dict(seed=seed, spread=spread, model="baseline",
                         val_acc=accuracy(clf.predict(task.x_val), task.y_val)))
        lc = LogHDClassifier(enc, 30, alphabet=2, num_bundles=5).fit(x_tr, y_tr, seed=seed)
        rows.append(dict(seed=seed, spread=spread, model="loghd",
                         val_acc=accuracy(lc.predict(task.x_val), task.y_val)))
    return rows


def run_refinement(seed: int, epochs: int, T: int) -> list[dict]:
    """Paper Alg.1 step 5 refinement on the hard shift task (k=2, n=min)."""
    task = make_domain_shift_task(seed=seed)
    x_tr, y_tr = task.x_train, task.y_train
    enc = ProjectionEncoder(x_tr.shape[1], BASE_DIM, seed=seed)
    enc.fit_feature_range(x_tr)
    lc = LogHDClassifier(enc, 30, alphabet=2, num_bundles=5)
    lc.fit(x_tr, y_tr, refine_epochs=T, seed=seed)
    return [dict(seed=seed, refine_epochs=T,
                 val_acc=accuracy(lc.predict(task.x_val), task.y_val),
                 avg_ood_acc=sum(accuracy(lc.predict(xt), yt) for _, xt, yt in task.targets) / 3)]


def plot_curves(df: pd.DataFrame, task: str, out_png: Path) -> None:
    sub = df[df.seed == df.seed.min()]
    base = sub[sub.model == "baseline"]
    fig, axes = plt.subplots(1, 3 if task == "novel" else 2, figsize=(15, 4.2))
    cols = ["id_acc", "avg_ood_acc"] if task == "shift" else ["id_acc", "nmi", "auroc"]
    titles = ["ID test acc", "avg OOD acc"] if task == "shift" else ["ID test acc", "novel NMI", "OOD-det AUROC"]
    for ax, col, title in zip(axes, cols, titles):
        for k in (2, 3):
            g = sub[(sub.model == "loghd") & (sub.k == k)].sort_values("n")
            ax.plot(g.n, g[col], marker="o", label=f"LogHD k={k}")
            g = sub[(sub.model == "feataxis") & (sub.k == k)].sort_values("n")
            ax.plot(g.n, g[col], marker="s", ls="--", label=f"feataxis (matched mem) k={k}")
        ax.axhline(base[col].mean(), ls=":", color="gray", label="baseline")
        ax.set_xlabel("bundles n")
        ax.set_ylabel(title)
        ax.set_title(f"{task}: {title}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_csweep(df: pd.DataFrame, out_png: Path) -> None:
    agg = df.groupby(["C", "model", "k"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ("val_acc", "avg_ood_acc"), ("ID val acc", "avg OOD acc")):
        base = agg[agg.model == "baseline"]
        ax.plot(base.C, base[col], "o-", color="gray", label="baseline (C prototypes)")
        for k in (2, 3):
            g = agg[(agg.model == "loghd") & (agg.k == k)]
            ax.plot(g.C, g[col], marker="o", label=f"LogHD k={k} (n=ceil log)")
        ax.set_xlabel("classes C")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_spread(df: pd.DataFrame, out_png: Path) -> None:
    agg = df.groupby(["spread", "model"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model, color in [("baseline", "gray"), ("loghd", "C0")]:
        g = agg[agg.model == model]
        ax.plot(g.spread, g.val_acc, marker="o", color=color, label=model)
    ax.set_xlabel("class separation (spread)")
    ax.set_ylabel("ID val acc")
    ax.set_title("LogHD clean accuracy tracks class separation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tasks", nargs="+", default=["shift", "novel"], choices=["shift", "novel"])
    ap.add_argument("--no-class-sweep", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    all_rows = []
    for task_name in args.tasks:
        for seed in range(args.seeds):
            all_rows.extend(run_grid(task_name, seed, args.epochs))
            print(f"[exp5a] done {task_name}/seed{seed}")
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "exp5_loghd.csv", index=False)
    for task_name in args.tasks:
        plot_curves(df[df.task == task_name], task_name, RESULTS / f"fig_exp5_{task_name}.png")

    if not args.no_class_sweep:
        crows = []
        for seed in range(args.seeds):
            crows.extend(run_class_sweep(seed, args.epochs))
        cd = pd.DataFrame(crows)
        cd.to_csv(RESULTS / "exp5_loghd_csweep.csv", index=False)
        plot_csweep(cd, RESULTS / "fig_exp5_csweep.png")

        srows = []
        for seed in range(args.seeds):
            srows.extend(run_spread_sweep(seed, args.epochs))
        sd = pd.DataFrame(srows)
        sd.to_csv(RESULTS / "exp5_loghd_spread.csv", index=False)
        plot_spread(sd, RESULTS / "fig_exp5_spread.png")

    rrows = []
    for seed in range(args.seeds):
        for T in (0, 20):
            rrows.extend(run_refinement(seed, args.epochs, T))
    rd = pd.DataFrame(rrows)
    rd.to_csv(RESULTS / "exp5_loghd_refine.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 50):
        g = df.groupby(["task", "model", "k", "n"]).mean(numeric_only=True).drop(columns=["seed"])
        cols = [c for c in ("val_acc", "id_acc", "avg_ood_acc", "nmi", "auroc") if c in g.columns]
        print(g[cols].round(3).to_string())


if __name__ == "__main__":
    main()
