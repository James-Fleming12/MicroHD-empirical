"""Exp1: sweep hyperspace dimensionality d and compare in-distribution
accuracy against generalization metrics (domain shift / novel-class discovery).

Shows whether ID accuracy saturates at a smaller d than the dimension needed
to preserve OOD robustness / novel-class structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.data import make_domain_shift_task, make_novel_class_task
from modules.hdc import HDCConfig, build_encoder
from modules.metrics import (
    accuracy,
    clustering_metrics,
    geometry_fidelity,
    ood_detection_auroc,
)
from modules.microhd import DIM_LADDER, cfg_seed
from modules.resources import memory_kb

RESULTS = Path(__file__).resolve().parents[1] / "results"

def train(cfg: HDCConfig, num_classes: int, x_train, y_train, seed: int, epochs: int):
    from modules.hdc import HDClassifier

    enc = build_encoder(cfg, x_train.shape[1], seed=cfg_seed(seed, cfg))
    enc.fit_feature_range(x_train)
    clf = HDClassifier(enc, num_classes).fit(x_train, y_train, epochs=epochs, seed=seed)
    return enc, clf

def eval_shift(encoding: str, dim: int, seed: int, epochs: int, levels: int) -> dict:
    task = make_domain_shift_task(seed=seed)
    n_cls = int(task.y_train.max()) + 1
    cfg = HDCConfig(encoding=encoding, dim=dim, levels=levels)
    enc, clf = train(cfg, n_cls, task.x_train, task.y_train, seed, epochs)
    row = {
        "encoding": encoding,
        "dim": dim,
        "seed": seed,
        "mem_kb": round(memory_kb(cfg, task.x_train.shape[1], n_cls), 1),
        "val_acc": accuracy(clf.predict(task.x_val), task.y_val),
        "src_acc": accuracy(clf.predict(task.x_test_src), task.y_test_src),
        "geom_fid": geometry_fidelity(enc, task.x_test_src),
    }
    for name, xt, yt in task.targets:
        row[f"acc_{name}"] = accuracy(clf.predict(xt), yt)
    return row

def eval_novel(encoding: str, dim: int, seed: int, epochs: int, levels: int) -> dict:
    task = make_novel_class_task(seed=seed)
    cfg = HDCConfig(encoding=encoding, dim=dim, levels=levels)
    enc, clf = train(cfg, task.num_known_classes, task.x_train, task.y_train, seed, epochs)
    h_nov = enc.encode(task.x_novel)
    cl = clustering_metrics(h_nov, task.num_novel_classes, task.y_novel.numpy(), seed=seed)
    s_id = clf.scores(task.x_test_id).max(dim=1).values.cpu().numpy()
    s_no = clf.scores(task.x_novel).max(dim=1).values.cpu().numpy()
    return {
        "encoding": encoding,
        "dim": dim,
        "seed": seed,
        "val_acc": accuracy(clf.predict(task.x_val), task.y_val),
        "id_acc": accuracy(clf.predict(task.x_test_id), task.y_test_id),
        "nmi": cl["nmi"],
        "ari": cl["ari"],
        "hungarian_acc": cl["hungarian_acc"],
        "auroc": ood_detection_auroc(s_id, s_no),
        "geom_fid": geometry_fidelity(enc, task.x_novel),
    }

def plot_curves(df: pd.DataFrame, task: str, out_png: Path) -> None:
    metrics_by_task = {
        "shift": (
            ["val_acc", "src_acc"],
            [c for c in df.columns if c.startswith("acc_")],
            ["geom_fid"],
        ),
        "novel": (
            ["val_acc", "id_acc"],
            ["nmi", "hungarian_acc", "auroc"],
            ["geom_fid"],
        ),
    }
    top, mid, extra = metrics_by_task[task]
    agg = df.groupby(["encoding", "dim"]).mean(numeric_only=True).reset_index()
    stds = df.groupby(["encoding", "dim"]).std(numeric_only=True).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0][0]
    for enc_kind, sub in agg.groupby("encoding"):
        s = stds[stds.encoding == enc_kind]
        ax.errorbar(sub.dim, sub.val_acc, yerr=s.val_acc, marker="o", label=f"val ({enc_kind})")
        ax.errorbar(sub.dim, sub.src_acc if "src_acc" in sub else sub.id_acc,
                    yerr=(s.src_acc if "src_acc" in s else s.id_acc),
                    marker="s", ls="--", label=f"test ({enc_kind})")
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("in-distribution accuracy")
    ax.set_title("ID accuracy saturates quickly")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for col in mid:
        for enc_kind, sub in agg.groupby("encoding"):
            s = stds[stds.encoding == enc_kind]
            label = col.replace("acc_", "") + (f" ({enc_kind})")
            ax.errorbar(sub.dim, sub[col], yerr=s[col], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_title("generalization keeps improving with d")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for enc_kind, sub in agg.groupby("encoding"):
        s = stds[stds.encoding == enc_kind]
        ax.errorbar(sub.dim, sub.geom_fid, yerr=s.geom_fid, marker="o", label=enc_kind)
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("spearman rho")
    ax.set_title("pairwise-geometry fidelity vs input space")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    if task == "shift":
        tgt_cols = [c for c in agg.columns if c.startswith("acc_")]
        agg = agg.assign(avg_ood=agg[tgt_cols].mean(axis=1))
        stds = stds.assign(avg_ood=stds[tgt_cols].mean(axis=1))
        for enc_kind, sub in agg.groupby("encoding"):
            s = stds[stds.encoding == enc_kind]
            ax.errorbar(sub.dim, sub.avg_ood, yerr=s.avg_ood, marker="o", label=f"avg OOD ({enc_kind})")
        ax.set_ylabel("avg target-domain acc")
    else:
        for enc_kind, sub in agg.groupby("encoding"):
            s = stds[stds.encoding == enc_kind]
            ax.errorbar(sub.dim, sub.nmi, yerr=s.nmi, marker="o", label=f"novel NMI ({enc_kind})")
        ax.set_ylabel("NMI of novel-class clusters")
    ax2 = ax.twinx()
    for enc_kind, sub in agg.groupby("encoding"):
        ax2.plot(sub.dim, sub.val_acc, marker=".", ls=":", color="gray", alpha=0.7)
    ax2.set_ylabel("ID val acc (dotted)", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_title(f"{'OOD' if task=='shift' else 'discovery'} quality vs ID saturation")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dims", type=int, nargs="+", default=DIM_LADDER)
    ap.add_argument("--levels", type=int, default=1024)
    ap.add_argument("--tasks", nargs="+", default=["shift", "novel"], choices=["shift", "novel"])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    eval_fns = {"shift": eval_shift, "novel": eval_novel}

    for task in args.tasks:
        rows = []
        for encoding in ("proj", "id"):
            for seed in range(args.seeds):
                for dim in args.dims:
                    row = eval_fns[task](encoding, dim, seed, args.epochs, args.levels)
                    rows.append(row)
                    print(f"[{task}] {encoding} d={dim} seed={seed}: "
                          + ", ".join(f"{k}={v:.3f}" for k, v in row.items()
                                      if isinstance(v, float)))
        df = pd.DataFrame(rows)
        csv = RESULTS / f"exp1_{task}.csv"
        df.to_csv(csv, index=False)
        plot_curves(df, task, RESULTS / f"fig_exp1_{task}.png")
        print(f"saved {csv}")

if __name__ == "__main__":
    main()
