"""Exp3: ID vs OOD analysis of the DPQ-HD post-training compression stages.

For an uncompressed projection-HDC workload (d=10k, fp32), applies each
stage in isolation and sweeps its knob:

  D) low-rank decomposition (SVD of trained P / fresh random factors)
  P) trailing-dimension pruning (keep first D' dims)
  Q) MSE-based PTQ of projection matrix + class HVs

and evaluates in-distribution accuracy against domain-shift robustness and
novel-class discovery quality. Also reports combined DPQ operating points.
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
from modules.dpqhd import (
    CompressedHDCModel,
    DecomposedEncoder,
    QuantizedEncoder,
    TruncatedEncoder,
    decompose_projection,
    dpq_memory_bits,
    mse_ptq,
    truncate_class_hvs,
)
from modules.hdc import HDClassifier, ProjectionEncoder
from modules.metrics import (
    accuracy,
    clustering_metrics,
    geometry_fidelity,
    ood_detection_auroc,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASE_DIM = 10000
# Effective rank of P2@P1 is capped by min(F, D) = num_features; the paper's
# larger ranks (256+) apply to inputs like MNIST with F=784.
RANKS = [8, 16, 24, 32, 48, 64]
KEEP_FRACS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
Q_BITS = [2, 3, 4, 6, 8]
DPQ_POINTS = [  # (rank, keep_frac, bits_p=bits_w) - escalating compression
    (64, 1.0, 4),
    (64, 0.5, 3),
    (32, 0.3, 3),
    (16, 0.2, 3),
    (8, 0.1, 2),
]


def make_task(name: str, seed: int):
    return make_domain_shift_task(seed=seed) if name == "shift" else make_novel_class_task(seed=seed)


def eval_model(task_name: str, task, encoder, class_hvs: torch.Tensor, seed: int) -> dict:
    model = CompressedHDCModel(encoder, class_hvs)
    row: dict = {"geom_fid": geometry_fidelity(encoder, task.x_val[:1000])}
    if task_name == "shift":
        row["val_acc"] = accuracy(model.predict(task.x_val), task.y_val)
        row["id_acc"] = accuracy(model.predict(task.x_test_src), task.y_test_src)
        tgt = []
        for name, xt, yt in task.targets:
            a = accuracy(model.predict(xt), yt)
            row[f"acc_{name}"] = a
            tgt.append(a)
        row["avg_ood_acc"] = sum(tgt) / len(tgt)
    else:
        row["val_acc"] = accuracy(model.predict(task.x_val), task.y_val)
        row["id_acc"] = accuracy(model.predict(task.x_test_id), task.y_test_id)
        h_nov = encoder.encode(task.x_novel)
        cl = clustering_metrics(h_nov, task.num_novel_classes, task.y_novel.numpy(), seed=seed)
        row.update(nmi=cl["nmi"], hungarian_acc=cl["hungarian_acc"])
        s_id = model.scores(task.x_test_id).max(dim=1).values.cpu().numpy()
        s_no = model.scores(task.x_novel).max(dim=1).values.cpu().numpy()
        row["auroc"] = ood_detection_auroc(s_id, s_no)
    return row


def run_task(task_name: str, seed: int, epochs: int) -> tuple[list[dict], list[dict]]:
    if task_name == "shift":
        task = make_task(task_name, seed)
        x_tr, y_tr = task.x_train, task.y_train
        n_cls = int(y_tr.max()) + 1
    else:
        task = make_task(task_name, seed)
        x_tr = torch.cat([task.x_train, task.x_val])
        y_tr = torch.cat([task.y_train, task.y_val])
        n_cls = task.num_known_classes
    f_dim = x_tr.shape[1]

    # uncompressed fp32 workload (paper's "Uncompressed" baseline)
    enc_base = ProjectionEncoder(f_dim, BASE_DIM, seed=seed)
    enc_base.fit_feature_range(x_tr)
    clf = HDClassifier(enc_base, n_cls).fit(x_tr, y_tr, epochs=epochs, seed=seed)
    w_full = clf.class_hvs.clone()

    rows: list[dict] = []

    def add(stage: str, knob_name: str, knob_val, encoder, w, **mem_kw):
        m = eval_model(task_name, task, encoder, w, seed)
        mem = dpq_memory_bits(f_dim, n_cls, BASE_DIM, **mem_kw)
        rows.append(dict(task=task_name, seed=seed, stage=stage, knob=knob_name,
                         value=knob_val, mem_kb=round(mem / 8 / 1024, 1),
                         compression_vs_fp=round(
                             dpq_memory_bits(f_dim, n_cls, BASE_DIM) / mem, 2), **m))

    add("none", "baseline", "fp32", enc_base, w_full)

    for r in RANKS:
        a, b = decompose_projection(enc_base.proj, r, "svd", seed=seed)
        add("D", "rank", r, DecomposedEncoder(a.to(enc_base.proj.device), b.to(enc_base.proj.device)),
            w_full.clone(), decomposition=True, rank=r)

    # rand decomposition: single-pass re-accumulation (paper's post-training
    # setting) and, as a reference, full retraining to isolate its effect.
    for r in RANKS:
        a, b = decompose_projection(enc_base.proj, r, "rand", seed=seed)
        enc_r = DecomposedEncoder(a.to(enc_base.proj.device), b.to(enc_base.proj.device))
        clf_r = HDClassifier(enc_r, n_cls).fit(x_tr, y_tr, epochs=0, seed=seed)
        add("D", "rank_rand", r, enc_r, clf_r.class_hvs.clone(), decomposition=True, rank=r)
    for r in RANKS:
        a, b = decompose_projection(enc_base.proj, r, "rand", seed=seed)
        enc_r = DecomposedEncoder(a.to(enc_base.proj.device), b.to(enc_base.proj.device))
        clf_r = HDClassifier(enc_r, n_cls).fit(x_tr, y_tr, epochs=epochs, seed=seed)
        add("D", "rank_rand_retrain", r, enc_r, clf_r.class_hvs.clone(),
            decomposition=True, rank=r)

    for frac in KEEP_FRACS:
        keep = max(1, int(round(frac * BASE_DIM)))
        add("P", "keep_frac", frac, TruncatedEncoder(enc_base, keep),
            truncate_class_hvs(w_full, keep), keep_dim=keep)

    for bp in Q_BITS:
        q_enc = QuantizedEncoder(enc_base, bp)
        w_q = mse_ptq(w_full, bp)
        add("Q", "bits", bp, q_enc, w_q, bits_p=bp, bits_w=bp)

    combos = []
    for rank, keep_frac, bits in DPQ_POINTS:
        keep = max(1, int(round(keep_frac * BASE_DIM)))
        a, b = decompose_projection(enc_base.proj, min(rank, f_dim), "svd", seed=seed)
        a, b = a[:, : min(rank, f_dim)], b[: min(rank, f_dim)]
        enc_c = DecomposedEncoder(a.to(enc_base.proj.device), b.to(enc_base.proj.device))
        enc_c = _truncate_decomposed(enc_c, keep)
        w_c = mse_ptq(truncate_class_hvs(w_full, keep), bits)
        m = eval_model(task_name, task, enc_c, w_c, seed)
        mem = dpq_memory_bits(f_dim, n_cls, BASE_DIM, decomposition=True,
                              rank=min(rank, f_dim), keep_dim=keep, bits_p=bits, bits_w=bits)
        combos.append(dict(task=task_name, seed=seed, stage="DPQ",
                           config=f"r{min(rank,f_dim)}-k{keep_frac:g}-b{bits}",
                           mem_kb=round(mem / 8 / 1024, 1),
                           compression_vs_fp=round(dpq_memory_bits(f_dim, n_cls, BASE_DIM) / mem, 2),
                           **m))
    return rows, combos


def _truncate_decomposed(enc: DecomposedEncoder, keep: int):
    """Truncate trailing dims of a decomposed encoder (prune applied after D)."""
    assert keep <= enc.dim
    return DecomposedEncoder(enc.mat_a[:keep].contiguous(), enc.mat_b.contiguous())


def plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    base = df[(df.stage == "none")].groupby("seed").mean(numeric_only=True)

    ood_col, ood_label = ("avg_ood_acc", "avg OOD acc") if df.task.iloc[0] == "shift" \
        else ("nmi", "novel-class NMI")

    panels = [
        ("D", [("rank", "svd"), ("rank_rand", "rand")], "Decomposition: rank r", lambda v: v),
        ("P", [("keep_frac", "post-training prune")], "Pruning: kept dims fraction", lambda v: v),
        ("Q", [("bits", "MSE PTQ")], "Quantization: bitwidth", lambda v: v),
    ]
    for col, (stage, series, title, _) in enumerate(panels):
        ax = axes[0][col]
        ax2 = axes[1][col]
        for i, (knob, label) in enumerate(series):
            sub = df[(df.stage == stage) & (df.knob == knob)].groupby("value")
            mean, std = sub.mean(numeric_only=True), sub.std(numeric_only=True)
            ax.errorbar(mean.index, mean.id_acc, yerr=std.id_acc, marker="o",
                        color=f"C{i}", label=f"ID ({label})")
            ax2.errorbar(mean.index, mean[ood_col], yerr=std[ood_col], marker="o",
                         color=f"C{i}", label=f"{ood_label} ({label})")
        ax.axhline(base.id_acc.mean(), ls=":", color="gray")
        ax2.axhline(base[ood_col].mean(), ls=":", color="gray")
        ax.set_xscale("log")
        ax2.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("ID test acc" if col == 0 else "")
        ax2.set_ylabel(ood_label if col == 0 else "")
        ax2.set_xlabel("sweep value (log)")
        ax.legend(fontsize=7)
        ax2.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
    fig.suptitle(f"DPQ-HD stages on '{df.task.iloc[0]}' task: ID accuracy vs generalization "
                 "(top row: ID; bottom row: OOD/discovery)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tasks", nargs="+", default=["shift", "novel"], choices=["shift", "novel"])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    all_rows, all_combos = [], []
    for task_name in args.tasks:
        for seed in range(args.seeds):
            rows, combos = run_task(task_name, seed, args.epochs)
            all_rows.extend(rows)
            all_combos.extend(combos)
            print(f"[exp3] done {task_name}/seed{seed}")

    df = pd.DataFrame(all_rows)
    combos = pd.DataFrame(all_combos)
    for name, d in (("exp3_dpq.csv", df), ("exp3_dpq_combos.csv", combos)):
        d.to_csv(RESULTS / name, index=False)
    for task_name in args.tasks:
        plot(df[df.task == task_name], RESULTS / f"fig_exp3_{task_name}.png")

    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print("=== DPQ combined operating points ===")
        print(combos.groupby(["task", "config"]).mean(numeric_only=True)
              .drop(columns=["seed"]).round(3).to_string())


if __name__ == "__main__":
    main()
