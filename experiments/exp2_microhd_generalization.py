"""Exp2: run the full MicroHD optimizer (accuracy-driven pruning) on each
synthetic task, then compare the optimized model against the d=10k baseline
on in-distribution accuracy AND generalization metrics.

The optimizer only ever sees the ID train/val split, exactly as in the
paper: acceptance is based solely on held-out accuracy. The question is
whether models that pass the ID accuracy constraint lose domain-shift
robustness / novel-class discovery quality relative to the baseline.
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
from modules.microhd import DIM_LADDER, LEVEL_LADDER, QUANT_LADDER, MicroHDOptimizer, cfg_seed
from modules.resources import memory_kb

RESULTS = Path(__file__).resolve().parents[1] / "results"
THRESHOLDS = (0.005, 0.01, 0.05)

def make_task(name: str, seed: int):
    return make_domain_shift_task(seed=seed) if name == "shift" else make_novel_class_task(seed=seed)

def splits(task, name: str):
    if name == "shift":
        return task.x_train, task.y_train, task.x_val, task.y_val, int(task.y_train.max()) + 1
    x = torch.cat([task.x_train, task.x_val])
    y = torch.cat([task.y_train, task.y_val])
    n_val = len(task.x_val)
    return x, y, x[-n_val:], y[-n_val:], task.num_known_classes

def fit_eval(cfg: HDCConfig, num_features: int, num_classes: int, x_train, y_train, seed: int, epochs: int):
    from modules.hdc import HDClassifier

    enc = build_encoder(cfg, num_features, seed=cfg_seed(seed, cfg))
    enc.fit_feature_range(x_train)
    clf = HDClassifier(enc, num_classes).fit(x_train, y_train, epochs=epochs, seed=seed)
    return enc, clf

def generalization_row(task_name: str, task, cfg: HDCConfig, seed: int, epochs: int) -> dict:
    """Retrain `cfg` and evaluate on the held-out test splits."""
    if task_name == "shift":
        x_tr, y_tr = task.x_train, task.y_train
        n_cls = int(y_tr.max()) + 1
    else:
        x_tr = torch.cat([task.x_train, task.x_val])
        y_tr = torch.cat([task.y_train, task.y_val])
        n_cls = task.num_known_classes
    enc, clf = fit_eval(cfg, x_tr.shape[1], n_cls, x_tr, y_tr, seed, epochs)

    row = {
        "val_acc": None,
        "geom_fid": geometry_fidelity(enc, x_tr[:2000]),
    }
    if task_name == "shift":
        row["id_acc"] = accuracy(clf.predict(task.x_test_src), task.y_test_src)
        tgt = [accuracy(clf.predict(xt), yt) for _, xt, yt in task.targets]
        for (name, _, _), a in zip(task.targets, tgt):
            row[f"acc_{name}"] = a
        row["avg_ood_acc"] = sum(tgt) / len(tgt)
    else:
        row["id_acc"] = accuracy(clf.predict(task.x_test_id), task.y_test_id)
        h_nov = enc.encode(task.x_novel)
        cl = clustering_metrics(h_nov, task.num_novel_classes, task.y_novel.numpy(), seed=seed)
        row.update(nmi=cl["nmi"], hungarian_acc=cl["hungarian_acc"])
        s_id = clf.scores(task.x_test_id).max(dim=1).values.cpu().numpy()
        s_no = clf.scores(task.x_novel).max(dim=1).values.cpu().numpy()
        row["auroc"] = ood_detection_auroc(s_id, s_no)
    return row

def run_one(task_name: str, encoding: str, seed: int, epochs: int, verbose: bool):
    task = make_task(task_name, seed)
    x_tr, y_tr, x_va, y_va, n_cls = splits(task, task_name)

    baseline_cfg = HDCConfig(encoding=encoding, dim=DIM_LADDER[-1], levels=LEVEL_LADDER[-1],
                             quant_bits=QUANT_LADDER[-1])
    cache: dict = {}

    base_row = generalization_row(task_name, task, baseline_cfg, seed, epochs)
    base_mem = memory_kb(baseline_cfg, x_tr.shape[1], n_cls)

    rows, traces = [], []
    for thr in THRESHOLDS:
        opt = MicroHDOptimizer(
            baseline_cfg, x_tr, y_tr, x_va, y_va,
            threshold=thr, epochs=epochs, seed=seed, eval_cache=cache, verbose=verbose,
        )
        res = opt.optimize()
        for step in res.steps:
            traces.append(dict(task=task_name, encoding=encoding, seed=seed, threshold=thr,
                               **step.__dict__))
        final_row = generalization_row(task_name, task, res.config, seed, epochs)
        final_mem = memory_kb(res.config, x_tr.shape[1], n_cls)
        common = dict(
            task=task_name, encoding=encoding, seed=seed, threshold=thr,
            model_type="microhd",
            dim=res.config.dim, levels=res.config.levels, quant_bits=res.config.quant_bits,
            mem_kb=round(final_mem, 1),
            compression_vs_baseline=round(base_mem / max(final_mem, 1e-9), 2),
            val_acc=res.final_val_acc,
            val_drop_vs_baseline=round(base_acc_of(cache, baseline_cfg, encoding) - res.final_val_acc, 4),
        )
        rows.append({**common, **{k: v for k, v in final_row.items() if k != "val_acc"}})

        # baseline row recorded once per threshold group for easy comparison
        bfull = dict(
            task=task_name, encoding=encoding, seed=seed, threshold=thr,
            model_type="baseline",
            dim=baseline_cfg.dim, levels=baseline_cfg.levels,
            quant_bits=baseline_cfg.quant_bits,
            mem_kb=round(base_mem, 1), compression_vs_baseline=1.0,
            **base_row,
        )
        bfull["val_acc"] = base_acc_of(cache, baseline_cfg, encoding)
        bfull["val_drop_vs_baseline"] = 0.0
        rows.append(bfull)

    df = pd.DataFrame(rows)
    return df, pd.DataFrame(traces)

def base_acc_of(cache: dict, baseline_cfg: HDCConfig, encoding: str) -> float:
    key = (baseline_cfg.encoding, baseline_cfg.dim, baseline_cfg.levels, baseline_cfg.quant_bits)
    return cache[key]

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["task", "encoding", "threshold", "model_type"]).mean(
        numeric_only=True
    ).reset_index()

def plot_summary(df: pd.DataFrame, out_png: Path) -> None:
    tasks = df.task.unique()
    fig, axes = plt.subplots(1, 3 * len(tasks), figsize=(6 * len(tasks), 4.2), squeeze=False)
    for j, task in enumerate(tasks):
        sub = df[df.task == task]
        ood_col = "avg_ood_acc" if task == "shift" else "nmi"
        panels = [("mem_kb", "memory (KB)"), ("id_acc", "ID test acc"), (ood_col, "generalization metric")]
        for k, (col, label) in enumerate(panels):
            ax = axes[0][3 * j + k]
            width = 0.27
            xs = range(len(THRESHOLDS))
            for i, enc_kind in enumerate(("proj", "id")):
                base_val = sub[(sub.encoding == enc_kind) & (sub.model_type == "baseline")][col].mean()
                vals, dims_chosen = [], []
                for thr in THRESHOLDS:
                    r = sub[(sub.encoding == enc_kind) & (sub.model_type == "microhd") & (sub.threshold == thr)]
                    vals.append(r[col].mean() if len(r) else float("nan"))
                    dims_chosen.append(r.dim.mean() if len(r) else float("nan"))
                ax.bar([x + (i - 0.5) * width for x in xs], vals, width=width,
                       label=enc_kind if k == 0 else None)
                ax.axhline(base_val, ls=":", color=f"C{i}", alpha=0.9,
                           label="baseline" if k == 0 and i == 0 else None)
                if k == 0:
                    lo = sub[sub.encoding == enc_kind][col].min()
                    for x, dv in zip(xs, dims_chosen):
                        ax.annotate(f"d={dv:.0f}", (x + (i - 0.5) * width, lo),
                                    fontsize=7, ha="center", rotation=90)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([f"{t*100:.1f}%" for t in THRESHOLDS])
            ax.set_xlabel("MicroHD accuracy threshold")
            ax.set_title(f"{task}: {label}", fontsize=10)
            ax.grid(alpha=0.25, axis="y")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tasks", nargs="+", default=["shift", "novel"], choices=["shift", "novel"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    all_rows, all_traces = [], []
    for task_name in args.tasks:
        for encoding in ("proj", "id"):
            for seed in range(args.seeds):
                df, tr = run_one(task_name, encoding, seed, args.epochs, args.verbose)
                all_rows.append(df)
                all_traces.append(tr)
                print(f"[exp2] done {task_name}/{encoding}/seed{seed}")
    runs = pd.concat(all_rows, ignore_index=True)
    traces = pd.concat(all_traces, ignore_index=True)
    runs.to_csv(RESULTS / "exp2_runs.csv", index=False)
    traces.to_csv(RESULTS / "exp2_traces.csv", index=False)

    summary = summarize(runs)
    summary.to_csv(RESULTS / "exp2_summary.csv", index=False)
    plot_summary(summary, RESULTS / "fig_exp2_summary.png")

    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
