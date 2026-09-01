"""DPQ-HD: post-training compression of projection-encoded HDC.

Implements the three stages from the DPQ-HD paper (Pandey, Kulkarni, Wang,
Gungor, Ponzina & Rosing, 2025):

  D) Low-rank decomposition of the projection matrix P in R^{F x D} into
     P1 in R^{r x F}, P2 in R^{D x r}; encoding h = P2 @ (P1 @ x).
     Two variants:
       * "svd"  - truncated SVD of the *trained* P (pure post-training;
                  class HVs reused unchanged; exact when rank == F since
                  a Gaussian P with F<D has full rank F)
       * "rand" - fresh random factors (statistically identical to svd for
                  random P); requires a single-pass re-accumulation of the
                  class HVs (no retraining epochs)
  P) Pruning by truncation of the trailing dimensions (keep first D' dims)
     applied jointly to encoder rows and class HV entries.
  Q) MSE-based scale-search symmetric quantization (Algorithm 1), applied
     to both projection matrix and class HVs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .hdc import bipolar_sign


@torch.no_grad()
def mse_ptq(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Algorithm 1: MSE-based scale search for symmetric PTQ.

    Returns the dequantized tensor at the best candidate scale.
    Candidate scales are fractions {0.1..1.0} of s = t_max / q_max with
    q_max = 2^(b-1) - 1 and clip range [-2^(b-1), +2^(b-1)-1].
    """
    t = t.detach()
    q_max = 2 ** (bits - 1) - 1
    s0 = t.abs().max().clamp_min(1e-12) / q_max
    best_t, best_err = None, float("inf")
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s = frac * s0
        tq = torch.round(t / s).clamp(-(q_max + 1), q_max) * s
        err = ((tq - t) ** 2).mean().item()
        if err < best_err:
            best_err, best_t = err, tq
    return best_t


def decompose_projection(
    proj: torch.Tensor,
    rank: int,
    mode: str,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose P (D x F) into A (D x r) and B (r x F) so that encode is B.T(A.T? ...).

    Returns (A, B) such that h = sign((B @ (A.T @ x))) reproduces the
    composed matrix M = A @ B (mode="svd") or an independent random
    rank-r matrix (mode="rand").
    """
    d_dim, f_dim = proj.shape
    r = min(rank, min(d_dim, f_dim))
    if mode == "svd":
        u, s, vt = torch.linalg.svd(proj, full_matrices=False)
        a = u[:, :r] * s[:r]
        b = vt[:r]
    elif mode == "rand":
        g = torch.Generator().manual_seed(seed)
        b = torch.randn(r, f_dim, generator=g) / math.sqrt(f_dim)
        a = torch.randn(d_dim, r, generator=g) / math.sqrt(r)
    else:
        raise ValueError(mode)
    return a.contiguous(), b.contiguous()


class DecomposedEncoder:
    """Encoder whose effective projection has low rank."""

    def __init__(self, mat_a: torch.Tensor, mat_b: torch.Tensor) -> None:
        self.mat_a = mat_a  # (dim, r)
        self.mat_b = mat_b  # (r, num_features)
        self.dim = mat_a.shape[0]
        self.num_features = mat_b.shape[1]
        self.device = mat_a.device

    def fit_feature_range(self, x: torch.Tensor) -> None:
        pass  # API parity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], 4096):
            xb = x[i : i + 4096].to(self.mat_b.device, torch.float32)
            h1 = xb @ self.mat_b.T  # (n, r)
            h = h1 @ self.mat_a.T  # (n, dim)
            outs.append(bipolar_sign(h))
        return torch.cat(outs, dim=0)


class TruncatedEncoder:
    """Post-training pruning view: keep first `keep` dims of an existing encoder's P."""

    def __init__(self, base_encoder, keep: int) -> None:
        assert hasattr(base_encoder, "proj"), "pruning expects a ProjectionEncoder"
        self.base = base_encoder
        self.keep = keep
        self.dim = keep
        self.num_features = base_encoder.num_features
        self.device = base_encoder.device

    def fit_feature_range(self, x: torch.Tensor) -> None:
        pass  # API parity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode(x)[:, : self.keep]


class QuantizedEncoder:
    """Post-training MSE quantization of an existing projection matrix."""

    def __init__(self, base_encoder, bits_p: int) -> None:
        assert hasattr(base_encoder, "proj"), "quantization expects a ProjectionEncoder"
        self.base = base_encoder
        self.proj_q = mse_ptq(base_encoder.proj, bits_p)
        self.dim = base_encoder.dim
        self.num_features = base_encoder.num_features

    def fit_feature_range(self, x: torch.Tensor) -> None:
        pass  # API parity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], 4096):
            xb = x[i : i + 4096].to(self.proj_q.device, torch.float32)
            outs.append(bipolar_sign(xb @ self.proj_q.T))
        return torch.cat(outs, dim=0)


def truncate_class_hvs(class_hvs: torch.Tensor, keep: int) -> torch.Tensor:
    """Keep first `keep` entries of every class hypervector."""
    return class_hvs[:, :keep].clone()


class CompressedHDCModel:
    """Standalone inference model over arbitrary encoder + class HVs."""

    def __init__(self, encoder, class_hvs: torch.Tensor) -> None:
        self.encoder = encoder
        self.class_hvs = class_hvs

    @torch.no_grad()
    def scores(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        c = None
        outs = []
        for i in range(0, x.shape[0], batch_size):
            h = self.encoder.encode(x[i : i + batch_size]).float()
            if c is None:
                c = torch.nn.functional.normalize(self.class_hvs.to(h.device), dim=1).float()
            outs.append(h @ c.T)
        return torch.cat(outs, dim=0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.scores(x).argmax(dim=1)


def dpq_memory_bits(
    num_features: int,
    num_classes: int,
    dim: int,
    *,
    decomposition: bool = False,
    rank: int | None = None,
    keep_dim: int | None = None,
    bits_p: int = 32,
    bits_w: int = 32,
) -> int:
    """Memory accounting for DPQ-HD pipelines (bits).

    Uncompressed: P (F*D) + W (C*D) at 32-bit.
    After D:      factors (F*r + r*D) replace P (+ W C*D).
    After P:      factors truncated to keep_dim (+ W C*keep_dim).
    After Q:      stored at bitwidths bits_p / bits_w.
    """
    eff_dim = keep_dim if keep_dim is not None else dim
    if decomposition and rank is not None:
        r = min(rank, num_features, dim)
        p_elems = num_features * r + r * eff_dim
    else:
        p_elems = num_features * eff_dim
    w_elems = num_classes * eff_dim
    return p_elems * bits_p + w_elems * bits_w
