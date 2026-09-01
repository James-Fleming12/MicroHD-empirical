# MicroHD-empirical

An empirical study of **MicroHD** (Ponzina & Rosing, tinyML'24) applied to synthetic
workloads, testing the hypothesis:

> Accuracy-driven dimensionality reduction in HDC overfits to in-distribution
> validation accuracy and destroys the random-projection geometry that supports
> generalization — domain-shift robustness and novel-class discovery.

MicroHD and DPQ-HD are feature-axis compressors; the last section (exp5) also
stress-tests **LogHD** (Yun et al., arXiv:2511.03938), which compresses along
the *class axis* instead, to see whether a compressor that never touches the
encoder escapes the same failure modes.

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
  dpqhd.py      # DPQ-HD post-training stages: low-rank decomposition (SVD or
                # fresh random factors), trailing-dimension pruning views,
                # MSE-based scale-search PTQ (Alg. 1), DPQ memory model
  loghd.py      # LogHD class-axis compression: minimax-load k-ary codebook,
                # bundle hypervectors, activation profiles, optional bundle
                # refinement, plus bit-flip model-noise corruption + memory model
  data.py       # Synthetic tasks: domain shift (covariate shift + noise
                # inflation, graded severity) and novel-class discovery
                # (held-out classes from the same generative family)
  metrics.py    # OOD accuracy, k-means NMI/ARI/Hungarian accuracy,
                # OOD-detection AUROC (max-similarity score),
                # pairwise-geometry fidelity (spearman rho input vs encoded space)
tests/test_sanity.py   # sanity checks (level HV orthogonality, geometry
                       # preservation, classifier sanity, resource formulas,
                       # optimizer respects threshold, DPQ + LogHD component checks)
experiments/
  exp1_dimension_sweep.py       # sweep d, plot ID vs OOD curves
  exp2_microhd_generalization.py# full MicroHD runs vs baseline at 0.5/1/5% thresholds
  exp3_dpq_generalization.py    # ID-vs-OOD sweeps for DPQ-HD stages D / P / Q
  exp4_margins.py               # ID vs OOD similarity margins across d/rank/keep/bits
  exp4_nullspace.py             # D-stage: novel classes in row space vs null space
  exp4_shift_visibility.py      # D-stage: fraction of shift direction visible vs r/F
  exp5_loghd_generalization.py  # LogHD class-axis stress: matched-memory class-vs-
                                # feature axis, C- and separation-sweeps, refinement
  exp5_loghd_noise.py           # bit-flip robustness at matched memory (LogHD vs
                                # feature-axis vs hybrid) on ID and OOD accuracy
results/          # CSVs + PNG figures
```

## Usage

```bash
pip install -r requirements.txt
python -m tests.test_sanity
python experiments/exp1_dimension_sweep.py --seeds 3
python experiments/exp2_microhd_generalization.py --seeds 3
python experiments/exp3_dpq_generalization.py --seeds 3
python experiments/exp4_margins.py --seeds 3
python experiments/exp4_nullspace.py --seeds 3
python experiments/exp4_shift_visibility.py --seeds 3
python experiments/exp5_loghd_generalization.py --seeds 3
python experiments/exp5_loghd_noise.py --seeds 3
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

## DPQ-HD analysis (exp3, 3 seeds)

DPQ-HD (post-training compression: Decomposition -> Pruning -> Quantization,
no retraining) is analyzed stage by stage on the same tasks
(`results/exp3_dpq.csv`, `fig_exp3_{shift,novel}.png`). The uncompressed fp32
baseline (d=10k) has shift-task ID/OOD = 0.996/0.722 and novel NMI/AUROC =
0.965/0.984.

**D) Decomposition (rank r).** The destructive knob. Effective rank caps at
F=64 (rank(P2@P1) <= F), so the paper's rank-256 settings only apply to
inputs with many features (e.g., MNIST F=784).

| r | val acc | OOD avg | (rand+retrain) OOD avg |
|---|---|---|---|
| 32 | 0.872 (-12%) | 0.467 (-25 pts) | — |
| 48 | 0.961 (-3.6%) | 0.630 (-9 pts) | 0.487 |
| 64 (=F) | 0.998 (lossless*) | 0.722 | 0.546 |

*rank-64 SVD reconstructs P exactly (Gaussian P has rank F), matching the
baseline. Two subtleties:
* OOD degrades ~2x faster than ID at matched rank (e.g., r=48: ID -3.6% but
  OOD -9.2 pts).
* Fresh random factors are strictly worse than SVD even after full 30-epoch
  retraining (r=64: OOD 0.526 single-pass -> 0.546 retrained, vs 0.722 for
  SVD). Mechanism: SVD retains the original encoder's singular-value
  weighting (top directions amplified before sign()); isotropic fresh factors
  spread signal evenly and lose that SNR advantage.

Memory savings from D alone are small (<=3x): factors are still fp32.

**P) Pruning (keep D'/D dims, post-training).** The cleanest ID-vs-OOD split:

| keep | memory | val drop | ID test | OOD avg |
|---|---|---|---|---|
| 20% | /5x | 0.6% | 0.985 | 0.689 |
| 10% | /10x | 0.9% | 0.982 | 0.670 |
| 5% | /20x | 1.5% | 0.976 | **0.612 (-11 pts)** |

A MicroHD-style calibration on 128 ID samples would accept 5% pruning (~1.5%
val drop) while domain-shift robustness collapses. Same failure mode as
MicroHD's dimension search, but achieved without any retraining.

**Q) Quantization (MSE scale-search PTQ).** Essentially free:

| bits_p=w | memory | val acc | OOD avg |
|---|---|---|---|
| 2 | /16x | 0.994 | 0.721 |
| 4 | /8x | 0.997 | 0.722 |

Even 2-bit MSE-PTQ costs ~nothing in OOD despite 39% relative RMSE in P:
the sign() nonlinearity absorbs projection-matrix quantization noise, and
coarse class-HV levels preserve cosine ordering. Contrast with exp2's naive
uniform-over-max quantization, where q=2 collapsed ID accuracy to 0.60 — how
you quantize matters more than how many bits.

**Combined DPQ operating points** (`results/exp3_dpq_combos.csv`):

| config | memory | val acc | OOD/NMI vs baseline |
|---|---|---|---|
| r64-k50%-b3 | 174 KB (/21x) | 0.996 | shift OOD -0.4 pts; novel NMI 0.967 (>= baseline) |
| r64-k100%-b4 | 383-461 KB (/8x) | ~1.00 | no measurable loss |
| r32-k30%-b3 | 51-69 KB (/53-59x) | 0.85-0.88 | OOD -27 pts; NMI 0.71 |

The safe operating region keeps rank = F (lossless decomposition) with
moderate pruning + aggressive MSE quantization: >20x compression with no
generalization loss. Pushing rank below F is what destroys generalization —
and a 128-sample ID calibration cannot detect it until ID itself degrades.

## LogHD analysis (exp5, 3 seeds)

**LogHD** (Yun et al., arXiv:2511.03938) compresses along the *class axis*,
the dimension the previous sections leave untouched. Instead of shrinking `d`
or quantizing the encoder, it keeps the full-dimensional encoder and replaces
the `C` class prototypes with `n ≈ ⌈log_k C⌉ + ε` bundle hypervectors plus `C`
activation profiles, cutting classifier memory from `O(CD)` to `O(D log_k C)`
(`modules/loghd.py`, `results/exp5_*.csv`). We stress-test it on the same two
tasks at matched classifier memory (`fig_exp5_{shift,novel}.png`).

**The hypothesis was half right.** Because the encoder is untouched, everything
that rides on the encoder *is* preserved: novel-class clustering NMI is
identical to baseline (0.966 vs 0.966), and pairwise-geometry fidelity is
unchanged. But the classifier LogHD substitutes — n-bundle cosines + nearest
activation-profile decoding — is far weaker than C-prototype cosine decoding on
these tasks, and this drags down ID accuracy, domain-shift OOD accuracy, and
OOD-detection AUROC. The class axis is *not* a free lunch; it has its own
superposition-interference bottleneck.

### Clean accuracy collapses with class count C

Matched-memory domain-shift task (baseline ID/OOD = 0.996/0.722):

| model | k | n | mem | ID test | avg OOD |
|---|---|---|---|---|---|
| baseline | — | — | 1.0 | 0.996 | 0.722 |
| feataxis (matched) | 2 | 5 | 0.167 | 0.995 | 0.685 |
| LogHD | 2 | 5 | 0.167 | 0.439 | 0.240 |
| LogHD | 2 | 7 | 0.234 | 0.587 | 0.264 |
| LogHD | 3 | 4 | 0.134 | 0.338 | 0.161 |
| LogHD | 3 | 6 | 0.201 | 0.501 | 0.231 |

At the same memory budget, feature-axis pruning keeps ID ≈ 0.995 and OOD
≈ 0.69 (the class prototypes are intact, just shorter); LogHD loses 40+ points
of ID and OOD. Novel-class task (baseline ID/NMI/AUROC = 0.978/0.966/0.984):
LogHD keeps NMI = 0.966 exactly but ID falls to 0.38–0.59 and OOD-detection
AUROC to 0.56–0.60 (vs 0.983 for feature-axis).

The degradation scales with class count (`exp5_loghd_csweep.csv`,
`fig_exp5_csweep.png`):

| C | baseline ID val | LogHD (k=2, n=⌈log₂C⌉) ID val |
|---|---|---|
| 5 | 1.000 | 0.920 |
| 12 | 0.999 | 0.700 |
| 20 | 0.998 | 0.553 |
| 30 | 0.997 | 0.449 |

Each bundle superposes ~C/2 prototypes, so the class-discriminative signal per
activation coordinate is diluted by superposition cross-talk, and the decoder
gets only `n ≈ log C` coordinates to separate C classes. The activation-space
margin (nearest-wrong-profile distance minus own-profile distance) is ≈ 0 even
on **training data** (mean −0.006 ID, −0.017 OOD), versus the 26σ/7.5σ cosine
margins of the prototype classifier. Redundant bundles (ε up to +2) and the
paper's bundle refinement (T=20, even with profiles re-estimated) recover only
a few points — the bottleneck is the R^n activation-space SNR, not bundle
count. Refinement alone with the paper's fixed profiles *hurts* (0.449 → 0.313).

The regime where LogHD ≈ baseline is well-separated classes
(`exp5_loghd_spread.csv`): at C=30, ID val recovers 0.449 → 0.667 → 0.884 →
0.988 as class spread grows 0.7 → 1.0 → 1.5 → 2.5. The paper's real datasets
(ISOLET, UCIHAR, PAGE, PAMAP2) sit in this easier regime; the synthetic
domain-shift family is deliberately overlapping, so it exposes the
superposition floor.

### Model-noise robustness is inverted at matched memory (exp5b)

The paper's headline is that class-axis compression sustains bit flips 2.5–3×
longer than feature-axis compression at equal memory, because D is preserved.
We test this on the C=12 shift task (UCIHAR-like regime) with per-bit flips on
the quantized stored state (`exp5_loghd_noise.py`): flip probability `p` at
which ID accuracy drops to 50% of clean (`p_id50`, higher = more robust):

| model | mem | clean ID | bits=2 | bits=4 | bits=8 |
|---|---|---|---|---|---|
| full-D baseline | 1.0 | 0.999 | 0.408 | 0.409 | 0.414 |
| feataxis (k2 n5 matched) | 0.42 | 0.999 | 0.403 | 0.408 | 0.414 |
| feataxis (k3 n4 matched) | 0.33 | 0.999 | 0.399 | 0.413 | 0.412 |
| LogHD (k2 n5) | 0.42 | 0.739 | 0.032 | 0.134 | 0.069 |
| LogHD (k3 n4) | 0.33 | 0.695 | 0.030 | 0.063 | 0.084 |
| hybrid (LogHD + 50% prune) | 0.21 | 0.736 | 0.032 | 0.131 | 0.068 |

The ordering is **reversed**: feature-axis (and full-D) prototypes tolerate
`p ≈ 0.40` at every bitwidth, while LogHD collapses at `p ≈ 0.03–0.13` —
roughly 3–13× *less* fault tolerance, not 2.5–3× more. The same holds for OOD
accuracy under flips (`p_ood90`, `fig_exp5_noise.png`).

Why: robustness in HDC comes from *averaging*. The C-prototype classifier
averages each query against C independent D-length vectors, so a bit flip in
any one prototype coordinate is a 1/√D relative perturbation of a cosine. LogHD
also averages over D per bundle, but the decision is then made on only
`n ≈ log C` activation coordinates — every bundle/profile corruption shows up
at full strength in one of the n values the decoder trusts, and the signal per
coordinate is already small (superposition). There is no class-axis redundancy
left to average over. This is the mirror image of the paper's premise: keeping
D preserves the *encoder* geometry (NMI, geometry fidelity), but the class-axis
classifier has already spent its redundancy on the codebook.

### The blind spot, in one line

LogHD's encoder-side generalization is genuinely preserved (novel-class
structure, geometry), but its classifier-side performance — ID accuracy,
domain-shift robustness, OOD detection, and bit-flip tolerance — is gated by a
low-dimensional activation space whose SNR is set by the data's class
separation, not by `D`. A stress test that only looks at encoder-side metrics
(e.g. NMI) would approve it; one that looks at the classifier it actually
deploys would not.

## Why compression breaks generalization (theory)

Generalization in HDC is a *geometric* guarantee, and all three knobs —
dimensionality `d`, rank `r`, and bitwidth `q` — cut the same three
quantities:

1. **Expressivity of the encoded space.** The encoder is

   $$h(x) = \mathrm{sign}(x P^\top) \in \{\pm 1\}^d$$

   It can output `2^d` patterns, but the number of *mutually near-orthogonal*
   directions it can host grows only exponentially in `d` (packing capacity
   $\sim 2^{c d}$ for a fixed angle tolerance). Shrinking `d`, or forcing the
   pre-activation through a rank-r bottleneck, shrinks this set.

2. **Room for other orthogonal vectors.** Two random independent vectors in
   $\mathbb{R}^d$ have cosine with std $\sim 1/\sqrt{d}$; the *largest* cosine
   a fresh random vector achieves against `M` fixed prototypes concentrates at
   the interference floor

   $$\rho_{\max}(M,d) \approx \sqrt{\frac{2\ln M}{d}}$$

   A novel class is discoverable only if its within-class similarity stays
   above $\rho_{\max}$ against all `C` trained prototypes *and* the other
   novel classes. The floor rises as $\sqrt{\ln M / d}$ — at small `d` the
   space literally has no room left to insert new orthogonal directions.
   Equivalently, ε-separating `M` items requires

   $$d \gtrsim \frac{2 \ln M}{\varepsilon^2}$$

   (a JL-type count: dimensionality must grow like $\ln M / \varepsilon^2$).

3. **Margin SNR.** A perturbation of size $\delta$ in the representation
   flips exactly the points whose margin is $< \delta$. ID points sit on
   margins that OnlineHD retraining actively inflates (large); OOD/discovery
   points sit on the *intrinsic*, unoptimized margins (small), so the same
   $\delta$ costs an order of magnitude more OOD accuracy than ID accuracy.

ID accuracy only needs (1)–(3) for the `C` *trained* classes, which is why it
saturates at `d ≈ 512–1024`. Domain shift and novel-class discovery need them
for `M = C + K` items in the intrinsic geometry with thin margins — so the
$d \gtrsim (2\ln M)/\varepsilon^2$ requirement lands at 4096–10000 (exp1).
Every compression below that trades ID accuracy (cheap, visible to validation)
against these three quantities (expensive, invisible to it).

Every quantitative claim below is checked empirically in the
[Theory validation (exp4)](#theory-validation-exp4) section and by the sanity
tests in `tests/test_sanity.py`.

### MicroHD: what its changes do to the space

**Dimension d.** Shrinking `d` is a triple cut:

* *Packing.* Near-orthogonal capacity is exponential in `d`; each trained
  prototype consumes a slot, and the residual room for `K` novel directions
  collapses. This is the "no room for other orthogonal vectors" failure in
  its purest form.
* *Interference floor.* At `d=128` with `M=24` (14 known + 10 novel),
  $\rho_{\max} \approx \sqrt{2\ln 24 / 128} \approx 0.22$; at `d=10k` it is
  $\approx 0.025$, ~9x smaller.
  Novel clusters separated by a margin of ~0.1 at `d=10k` sit inside the
  noise floor at `d=128` → NMI collapses 0.96 → 0.79–0.81 (exp2).
* *JL distortion.* Random projections preserve pairwise distances to relative
  error $\varepsilon$ only for $d \gtrsim c \ln n / \varepsilon^2$. The
  effective $\varepsilon$ for OOD points is much
  smaller than for retrained ID points, so the *required* `d` is an order of
  magnitude above where ID validation flattens. MicroHD's acceptance
  criterion measures ID val accuracy (large, retraining-inflated margins), so
  it keeps shrinking `d` past the point where the floor exceeds the OOD
  margin — the "optimal" dimension is too small by construction.

**Levels l (ID-level encoding).** Levels are a per-feature analog-to-digital
converter: the encoder map is piecewise-constant over `l` bins per feature.
Cutting `l` from 1024 to 16 coarsens each feature axis into 16 slots and
*raises the per-step bit-flip probability*

$$\rho \approx \frac{1 - 0.05^{1/(l-1)}}{2}$$

from `≈0.0015` at `l=1024` to `≈0.09` at `l=16`, so each bin-boundary
crossing now flips ~9% of the hypervector bits instead of ~0.15%. Domain
shift translates all feature values in one direction; with coarse bins a
large fraction of samples cross the same boundary together, coherently (not
randomly) warping the encoded geometry. The trained prototypes re-adapt to
the coarse map (ID survives), but the residual input-space margin that shift
robustness relied on is quantized away. Exp2 confirms this: even runs where
the d-probe was rejected lose OOD robustness from `l=1024→16` alone.

**Quantization q (uniform-over-max).** The step is set by the *tail*, not the
bulk. For the Gaussian projection $P \sim \mathcal{N}(0, 1/f)$,

$$\max|P| \approx \sqrt{2\ln(F d)}\,\sigma \approx 5\sigma, \qquad
  \delta = \frac{\max}{2^{q-1}-1},$$

so $\delta$ is `≈5σ` at `q=2` and `≈1.7σ` at `q=3` — the bulk of weights
(near $\pm\sigma$) is coarsely represented and small weights collapse to 0.
`sign()` is sensitive exactly near zero, so this coarsening preferentially
destroys the *marginal* threshold crossings — the thin OOD margins — while the
large trained margins survive. At `q=2` ($\delta \approx 5\sigma$) the map is
too coarse even for ID (exp2: 0.60). Deploy-time quantization of class HVs
`W` does the same to prototypes: `W` entries have a large accumulated range,
so a single max-scaled step coarsens the bulk.

### DPQ-HD: what each stage does to the space

**D — Decomposition (rank r).** The only stage that cuts *input-side*
expressivity.

$$h = \mathrm{sign}(x^\top P_2 P_1), \quad \mathrm{rank}(P_2 P_1) \le r$$

The encoding factors through $\mathbb{R}^r$, and any two inputs differing
only in the $(F-r)$-dim null space of the bottleneck map to *identical*
hypervectors — the encoder is non-injective and the classifier effectively
sees `r` input dimensions. Because `P` is Gaussian, its `F` singular values
all lie within ~±20% of a common value: truncating to `r < F` discards an
`O(1)` fraction `(F−r)/F` of the projection energy, not a tail.

* *Domain shift.* The shift direction is a random unit vector in
  $\mathbb{R}^F$; the expected fraction of its energy visible to the rank-r
  encoder is $r/F$ (its projection onto the `r`-dim row space). At `r=32`
  (F=64) half the shift is invisible; at `r=16`, 75% is discarded. The
  encoded class-region overlap therefore grows much faster than the
  input-space overlap → OOD falls ~2x faster than ID at matched rank (r=48:
  ID −3.6%, OOD −9 pts).
* *Novel classes.* A novel class whose separating direction lies in the null
  space is literally invisible to the classifier — discovery is impossible
  regardless of `d`.
* *SVD vs fresh factors.* SVD is the optimal (Eckart–Young) rank-r
  reconstruction and preserves as much of the *trained* geometry as possible,
  so the trained prototypes stay near-optimal without retraining. Fresh
  random factors replace the geometry with a statistically new projection;
  prototypes must be re-learned against it, and re-learning can only exploit
  the `r` retained directions, which — SVD concentrates energy into the top
  directions before `sign()` while isotropic factors spread it evenly — carry
  lower SNR. Retraining closes only part of the gap (r=64: 0.546 vs 0.722).
  At `r=F`, SVD is exact (`rank(P) = F`) — the lossless point.

**P — Pruning (keep first D' dims).** Not an input-side cut (all `F` features
are still read), but the same output-side capacity + SNR cut as MicroHD's
`d`-shrink: the hypervector is now $\{\pm 1\}^{D'}$ with near-orthogonal
capacity $\sim 2^{c D'}$ and similarity estimates averaged over `D'`
coordinates (noise $\sim 1/\sqrt{D'}$). The interference floor rises: at
`D'=500` (5% keep), `M=40` → $\rho_{\max} \approx 0.12$, vs $0.027$ at
`d=10k` — a ~4.5x smaller budget of room for every new direction. ID
survives because the trained margins are large enough that a *subset* of
coordinates still separates the known classes (val drop 1.5%); the shifted
points, whose margins are thin, now sit inside the noise floor of their own
similarity estimates. Note the geometry-fidelity metric (spearman rho, 0.888
at 5%) misses this entirely: truncation preserves pairwise *ordering*, while
the harm is in margin magnitude / SNR, which a rank correlation does not
measure.

**Q — Quantization (MSE-PTQ).** The benign stage — and the theory says why.
The MSE-optimal scale fits the *bulk* of the weight distribution, so the
quantization error $\eta$ is near-minimal and isotropic, and `sign()` is
scale-invariant: a pre-activation flips sign only when $|\mathrm{pre}| <
|\eta|$. For Gaussian pre-activations the flipped fraction is

$$\frac{1}{\pi}\,\frac{\sigma_{\eta}}{\sigma_{\mathrm{pre}}}$$

where $\sigma_\eta$ is the std of the quantization-induced pre-activation
perturbation and $\sigma_{\mathrm{pre}}$ the pre-activation std — negligible
at 2–4 bits. The ordering of pre-activations (and hence of cosine
similarities) is preserved, so expressivity, packing room and margins are all
intact (OOD 0.721 @2-bit vs 0.722 fp32). The exp2 disaster is not the bitcount
but the *scaling rule*: max-based uniform quantization sets $\delta$ from the
tail and coarsens the bulk; MSE-PTQ fits the bulk.

### The blind spot shared by both

MicroHD accepts a config when ID validation accuracy holds; a post-hoc DPQ
calibration on the same 128 ID samples would accept every destructive
operating point. That criterion cannot see any of the four harms because it
never probes them: it checks only the ~C trained classes (never novel
directions → blind to packing exhaustion), only points inside the training
margins (blind to the $1/\sqrt{d}$ noise floor), and only unshifted inputs
(blind to null-space and level-coarsening losses). The dimension, rank, and
levels that generalize are not the ones that maximize ID accuracy — and a
model that has already pruned them cannot un-prune.

## Theory validation (exp4)

The analytic claims of the previous section are checked directly
(`experiments/exp4_*.py`, `results/exp4_*.csv`, plus the new sanity tests).

### Margins are the mechanism (exp4a, `exp4_margins.csv`)

Mean (correct-class − best-wrong-class) similarity on the shift task, in
units of the ≈1σ cosine-noise floor (class prototypes are unit-normalized, so
the margin scale is comparable across knobs):

| knob | config | ID margin | OOD margin | ID acc | avg OOD |
|---|---|---|---|---|---|
| baseline | d=10k fp32 | 26.1 | 7.5 | 0.996 | 0.722 |
| dim | d=4096 | 16.6 | 4.6 | 0.995 | 0.710 |
| dim | d=512 | 5.5 | 1.0 | 0.990 | 0.621 |
| dim | d=128 | 2.2 | −0.1 | 0.954 | 0.487 |
| rank | r=48 | 19.7 | 3.9 | 0.977 | 0.630 |
| rank | r=32 | 12.2 | −1.1 | 0.892 | 0.467 |
| rank | r=16 | 2.0 | −8.2 | 0.592 | 0.252 |
| keep | 20% | 11.5 | 2.9 | 0.995 | 0.689 |
| keep | 10% | 8.0 | 1.9 | 0.993 | 0.670 |
| keep | 5% | 5.5 | 0.9 | 0.990 | 0.612 |
| bits | 4 | 25.9 | 7.4 | 0.996 | 0.722 |
| bits | 2 | 22.6 | 6.5 | 0.997 | 0.721 |

OOD margins collapse toward (and below) the noise floor while ID margins
shrink more slowly and stay positive — the "thin OOD margins" mechanism.
At rank 16 OOD sits at −8.2σ (deep in error) while ID still holds 0.59; at
d=128 OOD crosses zero (errors) while ID holds 0.95. MSE-PTQ leaves both
margins essentially intact, matching its near-lossless Q-stage results.

### Null-space blindness is causal (exp4b, `exp4_nullspace.csv`)

Novel classes whose separating directions lie in the *row space* of the
rank-r bottleneck stay discoverable (NMI ≈ 1.0); classes separated along the
*null space* collapse to chance (NMI ≈ 0.01) — even though raw input-space
clustering is ≈ 1.0 for both. The decomposition itself, not the data, makes
those directions invisible.

| rank | input NMI | row-space NMI | null-space NMI |
|---|---|---|---|
| 8 | 0.999 | 1.000 | 0.006 |
| 16 | 1.000 | 0.998 | 0.010 |
| 32 | 1.000 | 1.000 | 0.010 |
| 48 | 1.000 | 1.000 | 0.009 |
| 64 (=F) | 1.000 | 0.999 | — (no null space) |

### Shift visibility tracks r/F (exp4c, `exp4_shift_visibility.csv`)

The empirical fraction of the domain-shift direction visible to a rank-r
encoder is ≈ r/F, and OOD accuracy collapses as the visible fraction shrinks:

| rank | r/F | visible fraction | avg OOD |
|---|---|---|---|
| 8 | 0.125 | 0.107 | 0.168 |
| 16 | 0.250 | 0.230 | 0.252 |
| 24 | 0.375 | 0.361 | 0.366 |
| 32 | 0.500 | 0.516 | 0.467 |
| 48 | 0.750 | 0.725 | 0.630 |
| 64 | 1.000 | 1.000 | 0.722 |

### Analytic quantities (sanity tests)

`tests/test_sanity.py` now checks the load-bearing formulas directly:

* *Interference floor.* The max cosine of a fresh random vector against M
  prototypes scales as √(2 ln M / d): measured ratio 9.0 across d=128→10k vs
  the predicted 8.8.
* *Tail-scaling.* uniform-over-max quantization has 3–6x higher reconstruction
  MSE than MSE-PTQ at matched bits (2–4), confirming that the max-scaled step
  coarsens the bulk.
* *sign() absorption.* MSE-PTQ flips far fewer pre-activation signs than
  uniform-over-max at the same bitwidth, and the flip fraction scales linearly
  with the quantization-induced noise (measured 1.97 vs predicted 1.94 across
  bitwidths).

### Deviations for DPQ-HD

* The paper's decomposition says "randomly initialized" factors; we implement
  both SVD-of-trained-P (true post-training) and fresh random factors (+ a
  single-pass re-accumulation variant, plus a full-retraining reference).
* Rank sweeps restricted to <= F=64 because the effective rank of the composed
  matrix is capped by min(F,D); the paper's rank-256 configs require inputs
  with F >= rank.
* Quantization follows Algorithm 1 faithfully (scale candidates {0.1..s},
  clip to [-2^(b-1), 2^(b-1)-1], MSE-optimal choice); applied to both P' and W.
* Adaptive inference / early exit (paper Sec. 3.4) is orthogonal to this
  generalization study and not implemented.

## Deviations from the paper

* The paper reports hardware runtimes; here we use its Table 1 memory model
  and an elementwise-ops proxy for compute (only used to rank greedy steps).
* Level HV construction uses per-step bit-flip probability rho solving
  `(1-2*rho)^(l-1) ≈ 0.05` so first/last levels are quasi-orthogonal while
  adjacent levels stay similar.
* Class HV quantization is symmetric uniform rounding over the observed max
  magnitude (int-style grid), evaluated at inference time.