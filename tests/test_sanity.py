"""Sanity tests for the HDC framework. Run: python -m tests.test_sanity"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from modules.data import make_domain_shift_task
from modules.hdc import HDClassifier, HDCConfig, IDEncoder, ProjectionEncoder, bipolar_sign, build_encoder
from modules.metrics import clustering_metrics, geometry_fidelity, ood_detection_auroc
from modules.microhd import MicroHDOptimizer
from modules.resources import memory_bits, memory_kb

def test_loghd_codebook():
    from modules.loghd import build_codebook

    # every class gets a unique code; no all-zero (invisible) class; balanced loads
    B = build_codebook(30, 2, 5, seed=0)
    assert B.shape == (30, 5)
    assert len(B.unique(dim=0)) == 30, "codes must be unique"
    assert (B != 0).any(dim=1).all(), "all-zero code makes a class invisible"
    loads = B.float().sum(0)
    assert loads.max() - loads.min() <= 2, f"minimax-load balance violated: {loads}"

    B3 = build_codebook(30, 3, 4, seed=0)
    assert len(B3.unique(dim=0)) == 30 and (B3 != 0).any(dim=1).all()

def test_loghd_classifier_on_separable_data():
    from modules.loghd import LogHDClassifier
    from modules.metrics import accuracy

    task = make_domain_shift_task(seed=0, num_classes=10, spread=1.5)
    enc = ProjectionEncoder(task.x_train.shape[1], 4096, seed=0)
    enc.fit_feature_range(task.x_train)
    clf = HDClassifier(enc, 10).fit(task.x_train, task.y_train, epochs=10, seed=1)
    base = accuracy(clf.predict(task.x_test_src), task.y_test_src)
    lc = LogHDClassifier(enc, 10, alphabet=2, num_bundles=4).fit(task.x_train, task.y_train, seed=1)
    lh = accuracy(lc.predict(task.x_test_src), task.y_test_src)
    print(f"  easy task: baseline {base:.3f} vs LogHD {lh:.3f}")
    assert lh > 0.9, f"LogHD should be near-baseline on well-separated data, got {lh}"

def test_loghd_preserves_encoder_discovery():
    """Class-axis compression leaves the encoder untouched, so novel-class
    clustering quality (a function of the encoder only) must equal the baseline."""
    from modules.loghd import LogHDClassifier
    from modules.metrics import clustering_metrics

    task = make_domain_shift_task(seed=0, num_classes=10, spread=1.5)
    x_tr, y_tr = task.x_train, task.y_train
    enc = ProjectionEncoder(x_tr.shape[1], 4096, seed=0)
    enc.fit_feature_range(x_tr)
    base_nmi = clustering_metrics(enc.encode(task.x_test_src), 10,
                                  task.y_test_src.numpy(), seed=0)["nmi"]
    lc = LogHDClassifier(enc, 10, alphabet=2, num_bundles=4).fit(x_tr, y_tr, seed=0)
    lh_nmi = clustering_metrics(enc.encode(task.x_test_src), 10,
                                task.y_test_src.numpy(), seed=0)["nmi"]
    assert abs(base_nmi - lh_nmi) < 1e-9

def test_loghd_memory_model():
    from modules.loghd import loghd_memory_bits, matched_feature_keep

    C, D, n = 30, 10000, 5
    log_bits = loghd_memory_bits(C, D, n)
    assert log_bits == n * D * 32 + C * n * 32
    keep = matched_feature_keep(C, D, n)
    feataxis_bits = keep * C * 32
    assert abs(feataxis_bits - log_bits) <= 32 * C // 2, "matched memory budgets must agree"
    # class-axis memory is logarithmic in C while conventional is linear
    assert loghd_memory_bits(C, D, n) < 0.5 * C * D * 32

def test_loghd_bitflip():
    from modules.loghd import bitflip

    g = torch.Generator().manual_seed(0)
    t = torch.randn(50, 200, generator=g)
    assert torch.equal(bitflip(t, 0.0, 4, seed=1), t), "p=0 must be a no-op"
    # pre-quantize onto the b-bit grid so bit flips are the only source of change
    def on_grid(x, bits):
        qmax = 2 ** (bits - 1) - 1
        s = x.abs().max() / qmax
        return torch.round(x / s).clamp(-(qmax + 1), qmax) * s
    diff = {}
    for p in (0.05, 0.2, 0.5):
        diff[p] = (bitflip(on_grid(t, 4), p, 4, seed=1) != on_grid(t, 4)).float().mean().item()
    assert diff[0.05] < diff[0.2] < diff[0.5], f"flip rate must increase with p: {diff}"
    # higher bitwidth: a flipped bit perturbs the signal less (smaller fraction
    # of the stored value range on average).
    def damage(bits):
        base = on_grid(t, bits)
        s = t.abs().max()
        return (bitflip(base, 0.2, bits, seed=1) - base).abs().mean().item() / s
    assert damage(2) > damage(8), "coarser precision must suffer larger relative corruption"



def test_level_hv_structure():
    enc = IDEncoder(num_features=8, dim=2048, num_levels=64, seed=0)
    L = enc.level_hvs
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1)
    adj = cos(L[:-1], L[1:]).mean().item()
    ends = cos(L[:1], L[-1:]).item()
    assert adj > 0.8, f"adjacent level sim {adj}"   # ~= 1 - (1-eps**(1/(L-1)))
    assert abs(ends) < 0.15, f"endpoint sim {ends}"  # ~= eps_endpoint/2

def test_geometry_preservation_high_dim():
    torch.manual_seed(0)
    x = torch.randn(400, 32)
    enc = ProjectionEncoder(32, 4096, seed=0)
    fid = geometry_fidelity(enc, x, n_points=300)
    assert fid > 0.85, f"projection geometry fidelity {fid:.3f} too low at d=4096"
    enc_small = ProjectionEncoder(32, 64, seed=0)
    fid_small = geometry_fidelity(enc_small, x, n_points=300)
    assert fid_small < fid, "expected worse geometry at d=64"

def test_classifier_on_blobs():
    task = make_domain_shift_task(seed=0, train_per_class=200, target_per_class=50, num_classes=10, spread=1.0)
    cfg = HDCConfig(encoding="proj", dim=4096)
    enc = build_encoder(cfg, task.x_train.shape[1], seed=1)
    enc.fit_feature_range(task.x_train)
    clf = HDClassifier(enc, int(task.y_train.max()) + 1).fit(task.x_train, task.y_train, epochs=10, seed=1)
    acc_src = (clf.predict(task.x_test_src).cpu() == task.y_test_src).float().mean().item()
    acc_t1 = (clf.predict(task.targets[0][1]).cpu() == task.targets[0][2]).float().mean().item()
    print(f"  proj d=4096: src={acc_src:.3f} shift1={acc_t1:.3f}")
    assert acc_src > 0.8, f"source accuracy {acc_src}"

def test_quantization_effect():
    task = make_domain_shift_task(seed=0, train_per_class=150)
    accs = {}
    for bits in (16, 3):
        cfg = HDCConfig(encoding="id", dim=1024, levels=64, quant_bits=bits)
        enc = build_encoder(cfg, task.x_train.shape[1], seed=1)
        enc.fit_feature_range(task.x_train)
        clf = HDClassifier(enc, int(task.y_train.max()) + 1).fit(task.x_train, task.y_train, epochs=10, seed=1)
        accs[bits] = (clf.predict(task.x_test_src, deploy_bits=bits).cpu() == task.y_test_src).float().mean().item()
    print(f"  id d=1024: q16={accs[16]:.3f} q3={accs[3]:.3f}")
    assert accs[3] <= accs[16] + 0.02

def test_resource_model():
    # Table 1: ID-level total = d*(f + l + c*q); Proj total = d*q*(f + c)
    cfg_id = HDCConfig(encoding="id", dim=1000, levels=128, quant_bits=4)
    assert memory_bits(cfg_id, 30, 5) == 1000 * (30 + 128 + 5 * 4)
    cfg_p = HDCConfig(encoding="proj", dim=1000, quant_bits=8)
    assert memory_bits(cfg_p, 30, 5) == 1000 * 8 * 35
    assert memory_kb(cfg_id, 30, 5) > 0

def test_optimizer_shrinks_and_respects_threshold():
    task = make_domain_shift_task(seed=3, train_per_class=80, target_per_class=40,
                                  target_shift_norms=(6.0,), target_noises=(1.15,),
                                  num_classes=12)
    opt = MicroHDOptimizer(
        baseline_config=HDCConfig(encoding="proj", dim=4096),
        x_train=task.x_train, y_train=task.y_train,
        x_val=task.x_val, y_val=task.y_val,
        threshold=0.05, epochs=8, seed=0,
        ladders={"dim": [64, 256, 1024, 4096], "quant_bits": [2, 4, 8, 16]},
    )
    res = opt.optimize()
    from modules.resources import memory_kb
    m_before = memory_kb(res.baseline_config, task.x_train.shape[1], 10)
    m_after = memory_kb(res.config, task.x_train.shape[1], 10)
    drop = res.baseline_val_acc - res.final_val_acc
    print(f"  {res.baseline_config} -> {res.config}; mem {m_before:.0f}->{m_after:.0f} KB; "
          f"val {res.baseline_val_acc:.3f}->{res.final_val_acc:.3f}")
    assert m_after < m_before, "optimizer failed to compress"
    assert drop <= 0.05 + 1e-9, f"accepted config violates threshold by {drop}"

def test_novelty_metrics_smoke():
    g = torch.Generator().manual_seed(0)
    means = 8.0 * torch.nn.functional.one_hot(torch.arange(3), 64).float()
    y = torch.repeat_interleave(torch.arange(3), 200)
    task_data = means[y] + torch.randn(600, 64, generator=g)
    m = clustering_metrics(task_data, 3, y.numpy(), seed=0)
    assert m["hungarian_acc"] > 0.95
    auroc = ood_detection_auroc(np.random.rand(200) + 1.0, np.random.rand(200))
    assert 0.5 < auroc

def test_dpq_components():
    from modules.dpqhd import (
        CompressedHDCModel,
        DecomposedEncoder,
        TruncatedEncoder,
        decompose_projection,
        dpq_memory_bits,
        mse_ptq,
        truncate_class_hvs,
    )

    torch.manual_seed(0)
    F, D = 16, 2048
    P = torch.randn(D, F) / math.sqrt(F)
    x = torch.randn(64, F)

    # rank == F SVD decomposition reconstructs the encoder exactly
    a, b = decompose_projection(P, rank=F, mode="svd")
    enc_full = DecomposedEncoder(a, b)
    assert (enc_full.encode(x) == bipolar_sign(x @ P.T)).all()

    # lower-rank decomposition differs but stays valid bipolar
    a2, b2 = decompose_projection(P, rank=4, mode="svd")
    h4 = DecomposedEncoder(a2, b2).encode(x)
    assert set(h4.unique().tolist()) <= {-1.0, 1.0}

    # truncation at keep=dim equals baseline
    base_enc = ProjectionEncoder(F, D, seed=0)
    clf_hv = torch.randn(5, D)
    m_base = CompressedHDCModel(base_enc, clf_hv)
    m_trunc = CompressedHDCModel(TruncatedEncoder(base_enc, D), truncate_class_hvs(clf_hv, D))
    assert torch.allclose(m_base.scores(x), m_trunc.scores(x))

    # MSE PTQ error decreases with bitwidth
    T = torch.randn(1000, 32)
    errs = [((mse_ptq(T, b) - T) ** 2).mean().item() for b in (2, 3, 5)]
    assert errs[0] > errs[1] > errs[2], errs

    # memory model sanity: Q strictly below fp pipeline
    mem_fp = dpq_memory_bits(F, 5, D)
    mem_q = dpq_memory_bits(F, 5, D, bits_p=3, bits_w=3)
    mem_dpq = dpq_memory_bits(F, 5, D, decomposition=True, rank=8, keep_dim=D // 2,
                              bits_p=3, bits_w=3)
    assert mem_fp > mem_q > mem_dpq > 0


def _max_cos(M: int, d: int, trials: int = 300, seed: int = 0) -> float:
    """Mean over draws of the largest |cosine| a fresh random +/-1 vector
    achieves against M random +/-1 prototypes in R^d."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(trials):
        proto = torch.sign(torch.rand(M, d, generator=g) * 2 - 1)
        fresh = torch.sign(torch.rand(d, generator=g) * 2 - 1)
        vals.append((fresh @ proto.T).abs().max().item() / d)
    return float(np.mean(vals))

def test_interference_floor():
    """Interference floor: max cosine of a fresh vector vs M prototypes ~ sqrt(2 ln M / d).

    Validates the 'no room for other orthogonal vectors' claim of the theory
    section: the floor rises ~1/sqrt(d), so halving d costs sqrt(2) in the
    separation a novel direction can achieve.
    """
    M = 24
    f128 = _max_cos(M, 128)
    f10k = _max_cos(M, 10_000)
    pred_ratio = math.sqrt(10_000 / 128)  # analytic floor ratio across d
    emp_ratio = f128 / f10k
    print(f"  floor d=128 {f128:.3f}, d=10k {f10k:.3f}: emp ratio {emp_ratio:.2f}, pred {pred_ratio:.2f}")
    assert 0.6 * pred_ratio <= emp_ratio <= 1.4 * pred_ratio, \
        f"interference floor must scale like sqrt(2 ln M / d), got ratio {emp_ratio:.2f}"

def test_quant_scaling_tail():
    """uniform-over-max quantization (MicroHD) is far coarser than MSE-PTQ.

    The uniform step is set by max|P| ~ 5 sigma (the tail), so the bulk of a
    Gaussian projection is coarsely represented; MSE-PTQ fits the bulk.
    """
    from modules.dpqhd import mse_ptq
    from modules.hdc import uniform_quantize

    g = torch.Generator().manual_seed(0)
    P = torch.randn(4096, 64, generator=g) / 8.0
    for b in (2, 3):
        mse_u = ((uniform_quantize(P, b) - P) ** 2).mean().item()
        mse_m = ((mse_ptq(P, b) - P) ** 2).mean().item()
        print(f"  bits={b}: uniform MSE {mse_u:.4f} vs MSE-PTQ {mse_m:.4f}")
        assert mse_u > 3 * mse_m, \
            f"tail-scaled uniform quantization should be much coarser at {b} bits"

def test_mse_ptq_flip_fraction():
    """sign() absorbs MSE-PTQ noise: flips are few and scale linearly with sigma_eta.

    Contrast: tail-scaled uniform quantization (MicroHD) flips many more
    pre-activation signs at the same bitwidth — 'how you quantize matters'.
    """
    from modules.dpqhd import mse_ptq
    from modules.hdc import bipolar_sign

    F, D, n = 64, 4096, 2000
    g = torch.Generator().manual_seed(0)
    P = torch.randn(D, F, generator=g) / math.sqrt(F)
    x = torch.randn(n, F, generator=g)
    s = x @ P.T

    flips, sig_eta = {}, {}
    for b in (2, 3):
        sq = x @ mse_ptq(P, b).T
        flips[b] = (bipolar_sign(s) != bipolar_sign(sq)).float().mean().item()
        sig_eta[b] = (sq - s).std().item()
        qm = 2 ** (b - 1) - 1
        delta = P.abs().max() / qm
        Pu = torch.round(P / delta).clamp(-qm, qm) * delta
        flu = (bipolar_sign(s) != bipolar_sign(x @ Pu.T)).float().mean().item()
        print(f"  bits={b}: MSE-PTQ flip {flips[b]:.3f} vs uniform-over-max {flu:.3f}")
        assert flips[b] < flu, "MSE-PTQ must flip fewer coordinates than uniform-over-max"
    pred = sig_eta[2] / sig_eta[3]
    emp = flips[2] / flips[3]
    print(f"  flip scaling {emp:.2f} vs sigma-eta scaling {pred:.2f}")
    assert 0.7 * pred <= emp <= 1.3 * pred, \
        "flip fraction must scale linearly with the quantization-induced pre-activation noise"


if __name__ == "__main__":
    import time

    t0 = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[RUN ] {name}")
            fn()
            print(f"[PASS] {name}")
    print(f"All sanity tests passed in {time.time() - t0:.1f}s")
