"""Exp4a: similarity margins vs compression (domain-shift task).

The theory section claims OOD/discovery generalization is governed by the
decision margin (correct-prototype similarity minus best-wrong-prototype
similarity) on the *intrinsic*, unoptimized geometry, while ID accuracy is
carried by large, retraining-inflated margins. This experiment measures the
margin for each compression knob on the domain-shift task:

  dim  MicroHD dimension reduction (encoder retrained at each d)
  rank DPQ D, SVD decomposition (class HVs reused, no retraining)
  keep DPQ P, trailing-dimension pruning (no retraining)
  bits DPQ Q, MSE-PTQ (no retraining)

Prediction: ID margins stay large and roughly flat across every knob while
OOD margins collapse, reproducing the ~2x ID-vs-OOD accuracy gap at matched
compression. Quantization (MSE-PTQ) should preserve both margins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.data import make_domain_shift_task
from modules.dpqhd import (
    CompressedHDCModel,
    DecomposedEncoder,
    QuantizedEncoder,
    TruncatedEncoder,
    decompose_projection,
    mse_ptq,
    truncate_class_hvs,
)
from modules.hdc import HDClassifier, ProjectionEncoder
from modules.metrics import accuracy

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_FEATURES = 64
BASE_DIM = 10000
DIMS = [128, 512, 4096]
RANKS = [16, 32, 48, 64]
KEEP_FRACS = [0.05, 0.1, 0.2, 0.5, 1.0]
Q_BITS = [2, 3, 4]


def margin(enc, class_hvs, x, y) -> float:
    """Mean (correct-class cosine - best-wrong-class cosine); negative => errors."""
    h = enc.encode(x).float()
    cn = F.normalize(class_hvs.to(h.device), dim=1)
    s = h @ cn.T  # (n, C)
    n = y.shape[0]
    idx = torch.arange(n, device=s.device)
    sc = s[idx, y]
    so = s.clone()
    so[idx, y] = -float("inf")
    return (sc - so.max(dim=1).values).mean().item()


def eval_margins(enc, class_hvs, task) -> dict:
    model = CompressedHDCModel(enc, class_hvs)
    ood_m = [margin(enc, class_hvs, xt, yt) for _, xt, yt in task.targets]
    ood_a = [accuracy(model.predict(xt), yt) for _, xt, yt in task.targets]
    return {
        "id_margin": margin(enc, class_hvs, task.x_test_src, task.y_test_src),
        "ood_margin": float(np.mean(ood_m)),
        "id_acc": accuracy(model.predict(task.x_test_src), task.y_test_src),
        "avg_ood_acc": float(np.mean(ood_a)),
    }


def run_seed(seed: int, epochs: int) -> list[dict]:
    task = make_domain_shift_task(seed=seed)
    x_tr, y_tr = task.x_train, task.y_train
    n_cls = int(y_tr.max()) + 1

    enc_base = ProjectionEncoder(N_FEATURES, BASE_DIM, seed=seed)
    enc_base.fit_feature_range(x_tr)
    clf = HDClassifier(enc_base, n_cls).fit(x_tr, y_tr, epochs=epochs, seed=seed)
    w_full = clf.class_hvs.clone()

    rows = [dict(seed=seed, stage="dim", value=BASE_DIM, **eval_margins(enc_base, w_full, task))]

    for d in DIMS:
        enc = ProjectionEncoder(N_FEATURES, d, seed=seed)
        enc.fit_feature_range(x_tr)
        clf_d = HDClassifier(enc, n_cls).fit(x_tr, y_tr, epochs=epochs, seed=seed)
        rows.append(dict(seed=seed, stage="dim", value=d, **eval_margins(enc, clf_d.class_hvs.clone(), task)))

    for r in RANKS:
        a, b = decompose_projection(enc_base.proj, r, "svd", seed=seed)
        enc_r = DecomposedEncoder(a.to(enc_base.proj.device), b.to(enc_base.proj.device))
        rows.append(dict(seed=seed, stage="rank", value=r, **eval_margins(enc_r, w_full, task)))

    for frac in KEEP_FRACS:
        keep = max(1, int(round(frac * BASE_DIM)))
        enc_k = TruncatedEncoder(enc_base, keep)
        rows.append(dict(seed=seed, stage="keep", value=frac,
                         **eval_margins(enc_k, truncate_class_hvs(w_full, keep), task)))

    for bp in Q_BITS:
        enc_q = QuantizedEncoder(enc_base, bp)
        rows.append(dict(seed=seed, stage="bits", value=bp,
                         **eval_margins(enc_q, mse_ptq(w_full, bp), task)))
    return rows


def plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("dim", "dimension d", False),
              ("rank", "decomposition rank r", False),
              ("keep", "kept fraction (pruning)", False),
              ("bits", "MSE-PTQ bitwidth", True)]
    base = df[(df.stage == "dim") & (df.value == BASE_DIM)].mean(numeric_only=True)
    for ax, (stage, xlab, xlog) in zip(axes, panels):
        sub = df[df.stage == stage].groupby("value").mean(numeric_only=True)
        ax.plot(sub.index, sub.id_margin, marker="o", color="C0", label="ID margin")
        ax.plot(sub.index, sub.ood_margin, marker="s", color="C1", label="OOD margin")
        ax.axhline(base.id_margin, ls=":", color="C0", alpha=0.6)
        ax.axhline(base.ood_margin, ls=":", color="C1", alpha=0.6)
        ax.set_xlabel(xlab)
        if xlog:
            ax.set_xscale("log")
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_title(f"stage: {stage}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean similarity margin")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    all_rows = []
    for seed in range(args.seeds):
        all_rows.extend(run_seed(seed, args.epochs))
        print(f"[exp4a] done seed{seed}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "exp4_margins.csv", index=False)
    plot(df, RESULTS / "fig_exp4_margins.png")

    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print(df.groupby(["stage", "value"]).mean(numeric_only=True)
              .drop(columns=["seed"]).round(3).to_string())


if __name__ == "__main__":
    main()
