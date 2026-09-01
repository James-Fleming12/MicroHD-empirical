"""LogHD: logarithmic class-axis compression (Yun et al., arXiv:2511.03938).

Conventional HDC stores one prototype per class: O(C D) memory and C cosine
similarities per query. LogHD replaces the C prototypes with n >= ceil(log_k C)
bundle hypervectors and decodes in the induced n-dimensional activation space,
cutting classifier memory to O(D log_k C) while *preserving dimensionality D*.

Pipeline (Algorithm 1 of the paper):
  1. Class prototypes H_c = mean of encoded training examples, normalized.
  2. Capacity-aware k-ary codebook via minimax-load greedy selection with
     symbol weight g(s) = s/(k-1) and capacity surrogate U(w) = w^alpha.
  3. Bundles M_j = sum_c g(B_{c,j}) H_c, normalized.
  4. Activation profiles P_c = mean activation vector of class-c examples.
  5. Optional perceptron-style refinement of the bundles toward code-implied
     targets tau_j = 2*B/(k-1) - 1.
  Inference: A(x) = (delta(M_j, phi(x)))_j in R^n, y = argmin_c ||A - P_c||^2.

Also implements:
  * bitflip()      - model-noise corruption: quantize to `bits` and flip each
                     stored bit with probability p (used for the paper's
                     bit-flip robustness evaluation).
  * memory model   - classifier-side memory for class-axis (bundles+profiles)
                     vs the conventional C*D prototypes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .hdc import resolve_device


def build_codebook(
    num_classes: int,
    k: int,
    n: int,
    *,
    alpha: float = 1.0,
    max_pool: int = 4096,
    seed: int = 0,
) -> torch.Tensor:
    """Minimax-load greedy k-ary codebook (Eq. 2-3 of the paper).

    Assigns each class a unique length-n code drawn from {0..k-1}^n, greedily
    choosing the code that minimizes the worst-case updated per-bundle load
    (with a tiny uniform tie-breaking term for diversity). Returns B (C, n).
    """
    if k ** n < num_classes:
        raise ValueError(f"k^n = {k ** n} < C = {num_classes}: too few codes")
    g = torch.Generator().manual_seed(seed)
    pool = _code_pool(k, n, max_pool, g)
    # A class with the all-zero code contributes to *no* bundle and is
    # invisible to the decoder; drop it so every class is represented.
    pool = pool[~(pool == 0).all(dim=1)]
    if pool.shape[0] < num_classes:
        raise ValueError("candidate pool too small for C classes")

    weights = (pool.float() / (k - 1)) ** alpha  # (P, n) = U(g(s_j))
    loads = torch.zeros(n)
    B = torch.empty(num_classes, n, dtype=torch.long)
    avail = torch.ones(pool.shape[0], dtype=torch.bool)
    eps = 1e-4
    for c in range(num_classes):
        ids = torch.nonzero(avail).flatten()
        wc = weights[ids]
        obj = (loads[None, :] + wc).max(dim=1).values  # worst-case updated load
        tie = torch.rand(obj.shape, generator=g) * eps
        orig = int(ids[int((obj + tie).argmin())])
        B[c] = pool[orig]
        loads = loads + weights[orig]
        avail[orig] = False
    return B


def _code_pool(k: int, n: int, pool_size: int, generator: torch.Generator) -> torch.Tensor:
    """Candidate code pool: full enumeration when k^n is moderate, else a
    random subsample (paper: random pool suffices to flatten loads)."""
    total = k ** n
    if total <= pool_size:
        idx = torch.arange(total)
    else:
        idx = torch.randint(total, (pool_size,), generator=generator)
    codes = torch.empty(idx.shape[0], n, dtype=torch.long)
    rem = idx.clone()
    for j in range(n):
        codes[:, j] = rem % k
        rem //= k
    return codes


class LogHDClassifier:
    """LogHD classifier: bundles + activation profiles over an arbitrary encoder."""

    def __init__(
        self,
        encoder,
        num_classes: int,
        alphabet: int = 2,
        num_bundles: Optional[int] = None,
        *,
        device=None,
    ) -> None:
        self.encoder = encoder
        self.num_classes = num_classes
        self.alphabet = alphabet
        self.num_bundles = num_bundles or math.ceil(
            math.log(num_classes) / math.log(alphabet)
        )
        self.device = encoder.device if device is None else resolve_device(device)
        self.bundles: Optional[torch.Tensor] = None
        self.bundle_norm: Optional[torch.Tensor] = None
        self.profiles: Optional[torch.Tensor] = None
        self.codebook: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: float = 1.0,
        refine_epochs: int = 0,
        refine_lr: float = 3e-4,
        batch_size: int = 2048,
        max_pool: int = 4096,
        seed: int = 0,
    ) -> "LogHDClassifier":
        x = x.detach().to(self.device, torch.float32)
        y = y.detach().to(self.device, torch.long)
        k, nb = self.alphabet, self.num_bundles

        H = self._encode_cache(x, batch_size)  # (N, D) bipolar float

        # (1) class prototypes: mean of encoded examples, normalized.
        cnt = torch.bincount(y, minlength=self.num_classes).clamp_min(1).float()
        proto = torch.zeros(self.num_classes, H.shape[1], device=self.device)
        proto.index_add_(0, y, H)
        proto = F.normalize(proto / cnt.unsqueeze(1), dim=1)

        # (2) capacity-aware codebook; (3) weighted superposition of prototypes.
        B = build_codebook(self.num_classes, k, nb, alpha=alpha,
                           max_pool=max_pool, seed=seed).to(self.device)
        M = ((B.float() / (k - 1)).T @ proto)  # (n, D)
        M = F.normalize(M, dim=1)

        # (4) per-class expected activation profiles.
        scale = math.sqrt(self.encoder.dim)
        A = (H @ M.T) / scale  # (N, n) cosine activations in [-1, 1]
        P = torch.zeros(self.num_classes, nb, device=self.device)
        P.index_add_(0, y, A)
        P = P / cnt.unsqueeze(1)

        # (5) optional perceptron-style refinement of the bundles.
        if refine_epochs > 0:
            M = self._refine(H, y, B, M, k, refine_epochs, refine_lr, batch_size, seed)

        self.bundles = M
        self.bundle_norm = F.normalize(M, dim=1)
        self.profiles = P
        self.codebook = B
        return self

    def _encode_cache(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            outs.append(self.encoder.encode(x[i : i + batch_size]).to(torch.float32))
        return torch.cat(outs, dim=0)

    def _refine(self, H, y, B, M, k, epochs, lr, batch_size, seed) -> torch.Tensor:
        """Perceptron-style bundle refinement (Alg. 1 step 5).

        M_j <- M_j + eta (tau_j - A_j) phi(x), normalized after every update
        (the paper's "optionally followed by normalization"); we normalize
        every update for stability. tau_j = 2*B/(k-1) - 1 is the code-implied
        target activation. Vectorized over each batch's samples in a tight
        inner loop since normalization couples the updates.
        """
        g = torch.Generator().manual_seed(seed)
        tau = 2.0 * B[y].float() / (k - 1) - 1.0  # (N, n)
        n = H.shape[0]
        scale = math.sqrt(self.encoder.dim)
        Mn = F.normalize(M, dim=1)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                hb = H[idx]
                for s in range(hb.shape[0]):
                    a = hb[s] @ Mn.T / scale  # (n,) observed activations
                    Mn = Mn + lr * (tau[idx[s]] - a)[:, None] * hb[s][None, :]
                    Mn = F.normalize(Mn, dim=1)
        return Mn

    @torch.no_grad()
    def activation(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        scale = math.sqrt(self.encoder.dim)
        outs = []
        for i in range(0, x.shape[0], batch_size):
            h = self.encoder.encode(x[i : i + batch_size]).to(torch.float32)
            outs.append((h @ self.bundle_norm.T) / scale)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def distances(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Squared L2 distance of activation vectors to each class profile (N, C)."""
        P = self.profiles.to(self.device)
        outs = []
        for i in range(0, x.shape[0], batch_size):
            A = self.activation(x[i : i + batch_size])
            outs.append(((A[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        return torch.cat(outs, dim=0)

    def scores(self, x, **kw) -> torch.Tensor:
        return -self.distances(x, **kw)

    def predict(self, x, **kw) -> torch.Tensor:
        return self.distances(x, **kw).argmin(dim=1)

    def ood_score(self, x, **kw) -> torch.Tensor:
        """Confidence score (higher = more ID): negative closest-profile distance."""
        return -self.distances(x, **kw).min(dim=1).values


def activation_margin(clf: LogHDClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean (distance to nearest wrong profile - distance to own profile) in the
    activation space; positive => correct. Larger = more robust."""
    d = clf.distances(x)
    idx = torch.arange(d.shape[0], device=d.device)
    dc = d[idx, y.to(d.device)]
    dw = d.clone()
    dw[idx, y.to(d.device)] = float("inf")
    return float((dw.min(dim=1).values - dc).mean())


# ---------------------------------------------------------------------------
# Memory model (classifier-side, following the paper's C*D footprint).
# ---------------------------------------------------------------------------

def loghd_memory_bits(
    num_classes: int,
    dim: int,
    num_bundles: int,
    *,
    bits: int = 32,
    profile_bits: int = 32,
) -> int:
    """Classifier-side memory: n bundles (n*D) + C activation profiles (C*n)."""
    return num_bundles * dim * bits + num_classes * num_bundles * profile_bits


def loghd_memory_kb(
    num_classes: int, dim: int, num_bundles: int, *, bits: int = 32, profile_bits: int = 32
) -> float:
    return loghd_memory_bits(num_classes, dim, num_bundles, bits=bits,
                             profile_bits=profile_bits) / 8.0 / 1024.0


def matched_feature_keep(
    num_classes: int, dim: int, num_bundles: int, *, bits: int = 32, profile_bits: int = 32
) -> int:
    """Kept feature dims D' such that a conventional C-prototype model stores the
    same number of bits as the LogHD model (matched classifier-side memory)."""
    elems = loghd_memory_bits(num_classes, dim, num_bundles,
                              bits=bits, profile_bits=profile_bits)
    return max(1, round(elems / (bits * num_classes)))


def conventional_memory_bits(num_classes: int, dim: int, *, bits: int = 32) -> int:
    return num_classes * dim * bits


# ---------------------------------------------------------------------------
# Model-noise (bit-flip) corruption.
# ---------------------------------------------------------------------------

@torch.no_grad()
def bitflip(t: torch.Tensor, p: float, bits: int, seed: int = 0) -> torch.Tensor:
    """Corrupt stored fp32 parameters: quantize to `bits` symmetric ints and
    flip each bit of the two's-complement representation with probability p.
    Returns the dequantized (corrupted) tensor. p = 0 returns t unchanged."""
    if p <= 0.0 or bits >= 31:
        return t
    dev = t.device
    tc = t.detach().cpu()
    qmax = 2 ** (bits - 1) - 1
    scale = tc.abs().max().clamp_min(1e-12) / qmax
    q = torch.round(tc / scale).clamp(-(qmax + 1), qmax).long()
    u = q + 2 ** (bits - 1)  # unsigned two's complement
    g = torch.Generator().manual_seed(seed)
    for b in range(bits):
        m = (torch.rand(u.shape, generator=g) < p).long()
        u = u ^ (m << b)
    return ((u - 2 ** (bits - 1)).float() * scale).to(dev)
