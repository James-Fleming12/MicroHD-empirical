"""Exp5b: model-noise (bit-flip) robustness of LogHD vs feature-axis compression.

The LogHD paper's headline claim is that, at matched model memory, class-axis
compression (fewer, full-length vectors) tolerates stored-state bit flips far
better than feature-axis compression (shorter vectors): the paper reports
~2.5-3x higher sustainable bit-flip rates.

We test this on the domain-shift task at a moderate class count (C=12, in the
paper's UCIHAR regime) where LogHD's clean accuracy is still usable. Models are
compared at *matched classifier-side memory*:

  * LogHD k=2 n=5 (mem ~= 0.42x)   vs feature-axis pruned to D'=matched
  * LogHD k=3 n=4 (mem ~= 0.33x)   vs feature-axis pruned to D'=matched
  * full-D conventional baseline as the upper reference
  * hybrid (LogHD + 50% feature pruning) as an intermediate point

Stored parameters are quantized to `bits` and each bit of the two's-complement
representation is flipped with probability p (both bundles and activation
profiles for LogHD, per the paper's description). We report both in-distribution
test accuracy and domain-shift OOD accuracy under flips.
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

from modules.data import make_domain_shift_task
from modules.dpqhd import CompressedHDCModel, TruncatedEncoder, truncate_class_hvs
from modules.hdc import HDClassifier, ProjectionEncoder
from modules.loghd import (
    LogHDClassifier,
    bitflip,
    conventional_memory_bits,
    loghd_memory_bits,
    matched_feature_keep,
)
from modules.metrics import accuracy

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASE_DIM = 10000
NUM_CLASSES = 12
BITS = (2, 4, 8)
FLIP_PROBS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
# (k, n) LogHD configs, each compared to feature-axis at matched memory.
CONFIGS = [(2, 5), (3, 4)]
HYBRID_KEEP = 0.5  # keep fraction of dims for the hybrid LogHD+SparseHD point


def id_ood(task, predict_fn) -> tuple[float, float]:
    id_a = accuracy(predict_fn(task.x_test_src), task.y_test_src)
    ood_a = sum(accuracy(predict_fn(xt), yt) for _, xt, yt in task.targets) / len(task.targets)
    return id_a, ood_a


def corrupted_prototype_model(enc, w, p, bits, seed):
    return CompressedHDCModel(enc, bitflip(w, p, bits, seed=seed))


def corrupted_loghd(enc, lc, p, bits, seed, keep=None):
    Mb = lc.bundles if keep is None else lc.bundles[:, :keep]
    Mb = bitflip(Mb, p, bits, seed=seed)
    enc_c = TruncatedEncoder(enc, Mb.shape[1]) if keep is not None else enc
    l2 = LogHDClassifier(enc_c, lc.num_classes, alphabet=lc.alphabet, num_bundles=lc.num_bundles)
    l2.bundles = Mb
    l2.bundle_norm = torch.nn.functional.normalize(Mb, dim=1)
    l2.profiles = bitflip(lc.profiles, p, bits, seed=seed)
    l2.codebook = lc.codebook
    return l2


def run_seed(seed: int, epochs: int) -> list[dict]:
    task = make_domain_shift_task(seed=seed, num_classes=NUM_CLASSES)
    x_tr, y_tr = task.x_train, task.y_train
    enc = ProjectionEncoder(x_tr.shape[1], BASE_DIM, seed=seed)
    enc.fit_feature_range(x_tr)
    clf = HDClassifier(enc, NUM_CLASSES).fit(x_tr, y_tr, epochs=epochs, seed=seed)
    w_full = clf.class_hvs.clone()

    def eval_tag(tag, fn, mem_bits):
        rows = []
        for bits in BITS:
            for p in FLIP_PROBS:
                id_a, ood_a = fn(p, bits)
                rows.append(dict(seed=seed, tag=tag, bits=bits, p=p,
                                 mem_frac=round(mem_bits / conventional_memory_bits(NUM_CLASSES, BASE_DIM), 4),
                                 id_acc=id_a, ood_acc=ood_a))
        return rows

    rows = []
    rows += eval_tag("full",
                     lambda p, b: id_ood(task, lambda x: corrupted_prototype_model(enc, w_full, p, b, seed).predict(x)),
                     conventional_memory_bits(NUM_CLASSES, BASE_DIM))

    for k, n in CONFIGS:
        lc = LogHDClassifier(enc, NUM_CLASSES, alphabet=k, num_bundles=n).fit(x_tr, y_tr, seed=seed)
        keep = matched_feature_keep(NUM_CLASSES, BASE_DIM, n)
        enc_t = TruncatedEncoder(enc, keep)
        w_t = truncate_class_hvs(w_full, keep)
        mem = loghd_memory_bits(NUM_CLASSES, BASE_DIM, n)
        rows += eval_tag(f"loghd_k{k}_n{n}",
                         lambda p, b, lc=lc: id_ood(task, lambda x: corrupted_loghd(enc, lc, p, b, seed).predict(x)),
                         mem)
        rows += eval_tag(f"feataxis_k{k}_n{n}",
                         lambda p, b, et=enc_t, wt=w_t: id_ood(
                             task, lambda x: corrupted_prototype_model(et, wt, p, b, seed).predict(x)),
                         mem)

    lc_h = LogHDClassifier(enc, NUM_CLASSES, alphabet=2, num_bundles=5).fit(x_tr, y_tr, seed=seed)
    keep_h = max(1, int(HYBRID_KEEP * BASE_DIM))
    mem_h = loghd_memory_bits(NUM_CLASSES, keep_h, 5)
    rows += eval_tag("hybrid_k2_n5",
                     lambda p, b: id_ood(task, lambda x: corrupted_loghd(
                         enc, lc_h, p, b, seed, keep=keep_h).predict(x)),
                     mem_h)
    return rows


def p_cross(df_sub, col: str, frac: float) -> float | None:
    """Flip probability at which a method's (clean-scaled) accuracy drops below `frac`."""
    sub = df_sub.sort_values("p")
    clean = sub.iloc[0][col]
    if clean <= 0.0:
        return None
    target = clean * frac
    above = sub[sub[col] >= target]
    if len(above) == len(sub):
        return None  # never drops below
    if len(above) == 0:
        return 0.0
    last = above.iloc[-1]
    nxt = sub[sub.p > last.p].iloc[0]
    denom = nxt[col] - last[col]
    if denom == 0:
        return float(last.p)
    t = (target - last[col]) / denom
    return float(last.p + t * (nxt.p - last.p))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bits, tag), sub in df.groupby(["bits", "tag"]):
        rows.append(dict(bits=bits, tag=tag, mem_frac=sub.mem_frac.iloc[0],
                         clean_id=sub[sub.p == 0].id_acc.mean(),
                         p_id50=p_cross(sub, "id_acc", 0.5),
                         p_id90=p_cross(sub, "id_acc", 0.9),
                         p_ood90=p_cross(sub, "ood_acc", 0.9)))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_png: Path) -> None:
    bits = BITS[1]
    sub = df[df.bits == bits].groupby(["tag", "p"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row_i, (k, n) in enumerate(CONFIGS):
        for col_i, (col, title) in enumerate([("id_acc", "ID test acc"), ("ood_acc", "avg OOD acc")]):
            ax = axes[row_i][col_i]
            base = sub[sub.tag == "full"]
            ax.plot(base.p, base[col], ls=":", color="gray", label="full-D baseline")
            for tag, mstyle in [(f"loghd_k{k}_n{n}", "o"), (f"feataxis_k{k}_n{n}", "s")]:
                g = sub[sub.tag == tag]
                ax.plot(g.p, g[col], marker=mstyle, label=tag)
            if row_i == 0:
                g = sub[sub.tag == "hybrid_k2_n5"]
                ax.plot(g.p, g[col], marker="^", ls="--", label="hybrid (LogHD+50%)")
            ax.set_xlabel("bit-flip probability p")
            ax.set_ylabel(title)
            ax.set_title(f"k={k} n={n} (matched memory): {title}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle(f"Bit-flip robustness at matched memory (bits={bits})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        print(f"[exp5b] done seed{seed}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "exp5_loghd_noise.csv", index=False)
    plot(df, RESULTS / "fig_exp5_noise.png")

    summ = summarize(df)
    summ.to_csv(RESULTS / "exp5_loghd_noise_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
        piv = summ.pivot(index="tag", columns="bits", values="p_id50")
        print("\nflip probability at -50% ID acc (higher = more robust):")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
