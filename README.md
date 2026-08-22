# MicroHD-empirical

An empirical study of **MicroHD** (Ponzina & Rosing, tinyML'24) applied to synthetic
workloads, testing the hypothesis:

> Accuracy-driven dimensionality reduction in HDC overfits to in-distribution
> validation accuracy and destroys the random-projection geometry that supports
> generalization — domain-shift robustness and novel-class discovery.

## Layout

```
modules/
  hdc.py        # ID-level + non-linear projection encoders, OnlineHD-style
                # classifier (single pass + ep=30 retraining, lr=1), q-bit
                # deployment quantization of class HVs / projection matrix
  resources.py  # Memory model of Table 1 (ID: d*(f+l+cq), proj: d*q*(f+c))
                # and a compute-ops proxy for the greedy selector
  microhd.py    # MicroHD optimizer: greedy param selection + binary search
                # over admitted-value ladders; accept iff val acc drop <= threshold
  data.py       # Synthetic tasks: domain shift (covariate shift + noise
                # inflation, graded severity) and novel-class discovery
                # (held-out classes from the same generative family)
  metrics.py    # OOD accuracy, k-means NMI/ARI/Hungarian accuracy,
                # OOD-detection AUROC (max-similarity score),
                # pairwise-geometry fidelity (spearman rho input vs encoded space)
tests/test_sanity.py   # sanity checks (level HV orthogonality, geometry
                       # preservation, classifier sanity, resource formulas,
                       # optimizer respects threshold)
experiments/
  exp1_dimension_sweep.py       # sweep d, plot ID vs OOD curves
  exp2_microhd_generalization.py# full MicroHD runs vs baseline at 0.5/1/5% thresholds
results/          # CSVs + PNG figures
```

## Usage

```bash
pip install -r requirements.txt
python -m tests.test_sanity
python experiments/exp1_dimension_sweep.py --seeds 3
python experiments/exp2_microhd_generalization.py --seeds 3
```

Baseline hyper-parameters follow the paper: `d=10000`, `l=1024`, `q=16`,
retraining `ep=30`, `lr=1`. Ladders end at the baseline value; binary search
per parameter with greedy selection by largest memory saving.

## Findings (3 seeds, f=64)

### Exp1: dimension sweep (`results/exp1_*.csv`)

| d | proj: ID val | proj: OOD avg | proj: novel NMI |
|---|---|---|---|
| 256 | 0.982 | 0.593 | 0.914 |
| 1024 | 0.996 | 0.696 | 0.951 |
| 4096 | 0.996 | 0.743 | 0.962 |
| 10000 | 0.997 | 0.736 | 0.957 |

In-distribution accuracy saturates by d≈512-1024 while OOD accuracy and
novel-class NMI keep improving through d≈4096-10000. The dimension needed for
generalization is roughly one order of magnitude larger than the dimension
needed for ID accuracy.

Geometry fidelity (spearman rho between input-space and encoded-space
pairwise similarities) rises monotonically with d for projection encoding
(0.57→0.99), confirming that low-d compression breaks RP distance
preservation.

### Exp2: MicroHD vs baseline (`results/exp2_summary.csv`, `fig_exp2_summary.png`)

Projection encoding, domain-shift task (baseline avg-OOD = 0.725):

| threshold | chosen config | memory | val drop | ID test | avg OOD | ΔOOD |
|---|---|---|---|---|---|---|
| 0.5% | d≈683, q≈5.3 | 39 KB | 0.4% | 0.991 | 0.682 | −4.3 pts |
| 1% | d=512, q=3 | 17.6 KB | 0.6% | 0.991 | 0.676 | −4.9 pts |
| 5% | d=128, q=3 | 4.4 KB | 4.3% | 0.957 | 0.567 | −15.8 pts |

Novel-class task (baseline NMI = 0.964): MicroHD chooses d=128 at all
thresholds (val drop 0.2–3.5%) → NMI falls to ~0.79–0.81 (**−15 to −18 pts**) and
AUROC 0.984 → 0.92. For the ID-level encoder the effect is even stronger
(NMI 0.89 → 0.35–0.45).

Even runs where dim probes were rejected co-optimize levels/q down
(e.g., l=1024→16, q=16→2 at d=10k), which also costs several points of
OOD robustness.

### Interpretation

The acceptance criterion of MicroHD only sees ID validation accuracy, which
is far less demanding than the dimension required to preserve:
1. pairwise similarity structure (JL-type distortion grows as d shrinks),
2. margin/noise robustness under covariate shift,
3. cluster separation for unseen classes.

So the "optimal" dimension it finds is systematically too small for
generalization. A natural extension (hooked via `accept_fn`) would add an
OOD-proxy constraint to acceptance — e.g., require stability of predictions
under input perturbation or preserve a minimum effective rank / geometry
fidelity on unlabeled data.

## Deviations from the paper

* The paper reports hardware runtimes; here we use its Table 1 memory model
  and an elementwise-ops proxy for compute (only used to rank greedy steps).
* Level HV construction uses per-step bit-flip probability rho solving
  `(1-2*rho)^(l-1) ≈ 0.05` so first/last levels are quasi-orthogonal while
  adjacent levels stay similar.
* Class HV quantization is symmetric uniform rounding over the observed max
  magnitude (int-style grid), evaluated at inference time.