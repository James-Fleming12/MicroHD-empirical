"""Exp4c: how much of the domain shift a rank-r encoder can see.

The shift task translates all class means along a fixed unit direction u.
A rank-r bottleneck keeps an r-dimensional (random) row space, so the expected
fraction of u's energy visible to the encoder is r/F. This experiment:

  * measures the empirical projected fraction for the *actual* shift direction,
  * tracks OOD accuracy with the rank-r encoder (class HVs reused, no retrain).

Prediction: visible fraction ~ r/F; as r drops, less of the shift reaches the
classifier and OOD accuracy collapses.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.data import make_domain_shift_task
from modules.dpqhd import CompressedHDCModel, DecomposedEncoder, decompose_projection
from modules.hdc import HDClassifier, ProjectionEncoder
from modules.metrics import accuracy

RESULTS = Path(__file__).resolve().parents[1] / "results"
F = 64
BASE_DIM = 10000
RANKS = [8, 16, 24, 32, 48, 64]


def run_seed(seed: int, epochs: int) -> list[dict]:
    task = make_domain_shift_task(seed=seed)
    u = np.asarray(task.shift_dir, dtype=np.float64)
    x_tr, y_tr = task.x_train, task.y_train
    n_cls = int(y_tr.max()) + 1

    enc_base = ProjectionEncoder(F, BASE_DIM, seed=seed)
    enc_base.fit_feature_range(x_tr)
    clf = HDClassifier(enc_base, n_cls).fit(x_tr, y_tr, epochs=epochs, seed=seed)
    w_full = clf.class_hvs.clone()

    rows = []
    for r in RANKS:
        a, b = decompose_projection(enc_base.proj, r, "svd", seed=seed)
        enc_r = DecomposedEncoder(a.to(enc_base.proj.device), b.to(enc_base.proj.device))
        # SVD rows of b are orthonormal right singular vectors -> projection matrix b^T b
        ut = torch.from_numpy(u).float().to(b.device)
        visible = (b @ ut).norm().item() ** 2
        model = CompressedHDCModel(enc_r, w_full)
        ood = [accuracy(model.predict(xt), yt) for _, xt, yt in task.targets]
        rows.append(dict(
            seed=seed, rank=r, r_over_f=round(r / F, 3),
            visible_frac=float(visible),  # ||proj_rowspace u||^2 (u unit)
            avg_ood_acc=float(np.mean(ood)),
        ))
    return rows


def plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    agg = df.groupby("rank").mean(numeric_only=True)
    axes[0].plot(agg.index, agg.r_over_f, "k--", label="r/F (expectation)")
    axes[0].plot(agg.index, agg.visible_frac, marker="o", label="empirical ||proj u||^2")
    axes[0].set_xlabel("rank r")
    axes[0].set_ylabel("fraction of shift direction visible")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(agg.index, agg.avg_ood_acc, marker="s", color="C2")
    axes[1].set_xlabel("rank r")
    axes[1].set_ylabel("avg OOD acc")
    axes[1].grid(alpha=0.3)

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
        print(f"[exp4c] done seed{seed}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "exp4_shift_visibility.csv", index=False)
    plot(df, RESULTS / "fig_exp4_shift_visibility.png")

    with pd.option_context("display.width", 160):
        print(df.groupby("rank").mean(numeric_only=True)
              .drop(columns=["seed"]).round(3).to_string())


if __name__ == "__main__":
    main()
