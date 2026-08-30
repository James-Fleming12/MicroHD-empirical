"""Synthetic workloads for probing generalization of compressed HDC models.

Two setups:

1) Domain shift (covariate shift): class-conditional Gaussians with an
   anisotropic shared covariance. Target domains translate all class means
   along a fixed random direction and inflate within-class variance, with
   graded severity.

2) Novel-class discovery: same generative family, but some classes are held
   out of training entirely. Discovery quality is measured by clustering the
   encoded novel-class samples and matching clusters to ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

def _orthogonal_basis(f: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((f, f)))
    return q

def make_class_means(num_classes: int, num_features: int, spread: float, rng) -> np.ndarray:
    return rng.normal(0.0, spread, size=(num_classes, num_features))

def make_shared_cov(num_features: int, decay: float, rng) -> np.ndarray:
    """Random PSD covariance with power-law eigenvalue decay."""
    eig = np.power(np.arange(1, num_features + 1), -decay)
    eig = eig / eig.mean()
    basis = _orthogonal_basis(num_features, rng)
    return (basis * eig) @ basis.T, eig

def sample_gaussians(
    means: np.ndarray,
    chol: np.ndarray,
    n_per_class: int,
    shift: np.ndarray | None,
    noise_scale: float,
    rng,
) -> tuple[torch.Tensor, torch.Tensor]:
    k, f = means.shape
    z = rng.standard_normal((k * n_per_class, f)) @ chol.T * noise_scale
    x = np.repeat(means, n_per_class, axis=0) + z
    if shift is not None:
        x = x + shift
    y = np.repeat(np.arange(k), n_per_class)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))

@dataclass
class DomainShiftTask:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test_src: torch.Tensor
    y_test_src: torch.Tensor
    targets: list[tuple[str, torch.Tensor, torch.Tensor]]  # (name, X, y), severity ascending
    shift_dir: np.ndarray                                   # unit vector all means are translated along

@dataclass
class NovelClassTask:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor          # ID validation (used by MicroHD acceptance)
    y_val: torch.Tensor
    x_test_id: torch.Tensor      # ID test split (calibration for OOD detection)
    y_test_id: torch.Tensor
    x_novel: torch.Tensor        # pooled novel-class samples (never seen in training)
    y_novel: torch.Tensor        # ground-truth novel labels 0..K_nov-1
    num_known_classes: int
    num_novel_classes: int

def make_domain_shift_task(
    *,
    seed: int = 0,
    num_classes: int = 30,
    num_features: int = 64,
    spread: float = 0.7,
    decay: float = 0.4,
    train_per_class: int = 250,
    val_frac: float = 0.2,
    target_shift_norms: tuple[float, ...] = (6.0, 12.0, 18.0),
    target_noises: tuple[float, ...] = (1.15, 1.30, 1.45),
    target_per_class: int = 100,
) -> DomainShiftTask:
    assert len(target_shift_norms) == len(target_noises)
    rng = np.random.default_rng(seed)

    means = make_class_means(num_classes, num_features, spread, rng)
    cov, _ = make_shared_cov(num_features, decay, rng)
    chol = np.linalg.cholesky(cov)

    shift_dir = rng.standard_normal(num_features)
    shift_dir /= np.linalg.norm(shift_dir)

    def carve(x: torch.Tensor, y: torch.Tensor):
        n_total = x.shape[0]
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
        n_val = int(val_frac * n_total)
        # stratified carve-out
        val_idx = []
        counts: dict[int, int] = {}
        for i in perm.tolist():
            c = int(y[i])
            if counts.get(c, 0) < n_val // num_classes:
                val_idx.append(i)
                counts[c] = counts.get(c, 0) + 1
        val_mask = torch.zeros(n_total, dtype=torch.bool)
        val_idx_t = torch.tensor(sorted(val_idx))
        val_mask[val_idx_t] = True
        tr_mask = ~val_mask
        return x[tr_mask], y[tr_mask], x[val_mask], y[val_mask]

    x_tr_full, y_tr_full = sample_gaussians(means, chol, train_per_class, None, 1.0, rng)
    x_train, y_train, x_val, y_val = carve(x_tr_full, y_tr_full)

    x_src, y_src = sample_gaussians(means, chol, target_per_class, None, 1.0, rng)

    targets = []
    for shift_norm, noise in zip(target_shift_norms, target_noises):
        xt, yt = sample_gaussians(means, chol, target_per_class, shift_dir * shift_norm, noise, rng)
        targets.append((f"shift_{shift_norm:g}", xt, yt))

    return DomainShiftTask(
        name="domain_shift",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test_src=x_src, y_test_src=y_src,
        targets=targets,
        shift_dir=shift_dir,
    )

def make_novel_class_task(
    *,
    seed: int = 0,
    num_total_classes: int = 24,
    num_known_classes: int = 14,
    num_features: int = 64,
    spread: float = 0.55,
    decay: float = 0.4,
    train_per_class: int = 250,
    val_frac: float = 0.2,
    novel_test_per_class: int = 150,
    id_test_per_class: int = 100,
) -> NovelClassTask:
    assert num_known_classes < num_total_classes
    rng = np.random.default_rng(seed)

    means = make_class_means(num_total_classes, num_features, spread, rng)
    cov, _ = make_shared_cov(num_features, decay, rng)
    chol = np.linalg.cholesky(cov)

    known_means = means[:num_known_classes]

    x_tr_full, y_tr_full = sample_gaussians(known_means, chol, train_per_class, None, 1.0, rng)
    perm = torch.randperm(x_tr_full.shape[0], generator=torch.Generator().manual_seed(seed))
    n_val = int(val_frac * x_tr_full.shape[0])
    val_idx = perm[:n_val]
    mask = torch.zeros(x_tr_full.shape[0], dtype=torch.bool)
    mask[val_idx] = True
    x_train, y_train = x_tr_full[~mask], y_tr_full[~mask]
    x_val, y_val = x_tr_full[mask], y_tr_full[mask]

    x_id, y_id = sample_gaussians(known_means, chol, id_test_per_class, None, 1.0, rng)

    nov_means = means[num_known_classes:]
    x_nov, y_nov = sample_gaussians(nov_means, chol, novel_test_per_class, None, 1.0, rng)

    return NovelClassTask(
        name="novel_classes",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test_id=x_id, y_test_id=y_id,
        x_novel=x_nov, y_novel=y_nov,
        num_known_classes=num_known_classes,
        num_novel_classes=num_total_classes - num_known_classes,
    )
