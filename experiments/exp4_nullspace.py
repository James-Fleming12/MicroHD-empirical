"""Exp4b: DPQ decomposition creates null-space blindness for novel classes.

A rank-r decomposition h = sign(x^T P2 P1) with rank(P2 P1) <= r factors the
encoding through R^r: any two inputs differing only in the (F-r)-dim null
space of the bottleneck map to *identical* hypervectors. If a novel class is
separated from the others along such a direction it is literally invisible to
the classifier, no matter how large d is.

Construction: decompose a trained projection at rank r (SVD). Take the row
space of the bottleneck (top-r right singular vectors) and its null space
(the remaining singular vectors) as orthonormal direction sets. Build novel
classes whose means are separated along directions drawn from one set or the
other, then measure clustering recovery (NMI) of the *encoded* novel samples.

Controls:
  variant="input"  -> NMI of the raw input-space novel samples (must be high:
                      the classes are separable in input space for both alignments)
  variant="row"    -> novel classes separated in the row space of the bottleneck
  variant="nullspace"   -> novel classes separated in the null space of the bottleneck

At r=F the decomposition is lossless (null space empty) so "nullspace" degenerates
to "row"; below it, "nullspace" should collapse to chance clustering.
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

from modules.data import make_shared_cov, sample_gaussians
from modules.dpqhd import DecomposedEncoder
from modules.hdc import ProjectionEncoder
from modules.metrics import clustering_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"
F = 64
D = 4096
RANKS = [8, 16, 32, 48, 64]
C_KNOWN = 10
K_NOVEL = 6
N_PER = 150
AMP = 6.0          # separation of novel means along the chosen directions


def build_novel_means(rng, rank: int, align: str, enc):
    """Novel-class means anchored at 0, separated along `rank` directions
    chosen from the row space or the null space of the rank-r bottleneck."""
    u, s, vt = torch.linalg.svd(enc.proj, full_matrices=False)
    vt = vt.detach().cpu().numpy()
    row_basis = vt[:rank]                 # orthonormal rows spanning the row space
    null_basis = vt[rank:]                # orthonormal rows spanning the null space
    if align == "row":
        w = row_basis[:K_NOVEL]
    else:
        w = null_basis[:K_NOVEL]
    assert w.shape[0] == K_NOVEL, f"not enough {align}-space directions at rank {rank}"
    return AMP * w  # K_NOVEL x F novel means (unit separation directions)


def run_seed(seed: int, epochs: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cov, _ = make_shared_cov(F, 0.4, rng)
    chol = np.linalg.cholesky(cov)

    # train a projection encoder + classifier on known classes (realism:
    # the encoded space is populated by trained prototypes before novel data arrives)
    task_means = rng.normal(0.0, 0.55, size=(C_KNOWN, F))
    x_known, y_known = sample_gaussians(task_means, chol, 250, None, 1.0, rng)
    enc = ProjectionEncoder(F, D, seed=seed)
    enc.fit_feature_range(x_known)

    rows = []
    for rank in RANKS:
        a, b = decompose_svd(enc, rank)
        enc_r = DecomposedEncoder(a, b)

        for align in ("row", "nullspace"):
            if align == "nullspace" and rank >= F:
                continue  # null space empty at r=F; identical to "row"
            means = build_novel_means(rng, rank, align, enc)
            x_nov, y_nov = sample_gaussians(means, chol, N_PER, None, 1.0, rng)
            m_input = clustering_metrics(x_nov, K_NOVEL, y_nov.numpy(), seed=seed)
            h = enc_r.encode(x_nov)
            m_enc = clustering_metrics(h, K_NOVEL, y_nov.numpy(), seed=seed)
            rows.append(dict(seed=seed, rank=rank, variant=align,
                             input_nmi=m_input["nmi"],
                             nmi=m_enc["nmi"],
                             hungarian_acc=m_enc["hungarian_acc"]))
        # also record the raw-input-space control once per rank
        means = build_novel_means(rng, rank, "row", enc)
        x_nov, y_nov = sample_gaussians(means, chol, N_PER, None, 1.0, rng)
        m_input = clustering_metrics(x_nov, K_NOVEL, y_nov.numpy(), seed=seed)
        rows.append(dict(seed=seed, rank=rank, variant="input",
                         input_nmi=m_input["nmi"], nmi=m_input["nmi"],
                         hungarian_acc=float("nan")))
    return rows


def decompose_svd(enc, rank: int):
    from modules.dpqhd import decompose_projection
    return decompose_projection(enc.proj, rank, "svd", seed=0)


def plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, color in [("input", "gray"), ("row", "C0"), ("nullspace", "C3")]:
        sub = df[df.variant == variant].groupby("rank").mean(numeric_only=True)
        ax.plot(sub.index, sub.nmi, marker="o", color=color, label=variant)
    ax.axvline(F, color="gray", ls=":", lw=1)
    ax.annotate("r = F (lossless)", (F, 0.02), fontsize=8, ha="right")
    ax.set_xlabel("decomposition rank r")
    ax.set_ylabel("NMI of encoded novel clusters")
    ax.set_title("Null-space blindness of rank-r decomposition")
    ax.grid(alpha=0.3)
    ax.legend()
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
        print(f"[exp4b] done seed{seed}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "exp4_nullspace.csv", index=False)
    plot(df, RESULTS / "fig_exp4_nullspace.png")

    with pd.option_context("display.width", 160):
        print(df.groupby(["rank", "variant"]).mean(numeric_only=True)
              .drop(columns=["seed"]).round(3).to_string())


if __name__ == "__main__":
    main()
