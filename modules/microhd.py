"""MicroHD: accuracy-driven co-optimization of HDC hyper-parameters.

Greedy selection over tunable hyper-parameters with a binary search of each
parameter's admitted-value ladder (Sec. 4.2 of the paper). A candidate model
is retrained from scratch and accepted only if held-out accuracy stays within
a user-defined threshold of the baseline model.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable

import torch

from .hdc import HDClassifier, HDCConfig, build_encoder, resolve_device

DIM_LADDER = [64, 128, 256, 512, 1024, 2048, 4096, 10000]
LEVEL_LADDER = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
QUANT_LADDER = [2, 3, 4, 6, 8, 12, 16]

DEFAULT_LADDERS: dict[str, dict[str, list[int]]] = {
    "id": {"dim": DIM_LADDER, "levels": LEVEL_LADDER, "quant_bits": QUANT_LADDER},
    "proj": {"dim": DIM_LADDER, "quant_bits": QUANT_LADDER},
}

@dataclass
class OptStep:
    iteration: int
    param: str
    from_value: int
    to_value: int
    accepted: bool
    val_acc: float
    val_drop: float
    mem_kb_before: float
    mem_kb_after: float

@dataclass
class OptResult:
    config: HDCConfig
    baseline_config: HDCConfig
    baseline_val_acc: float
    final_val_acc: float
    steps: list[OptStep] = field(default_factory=list)

    @property
    def compression(self) -> float:
        return self.baseline_config.dim / max(self.config.dim, 1)

def cfg_seed(master: int, cfg: HDCConfig) -> int:
    """Deterministic per-config seed (common across sweeps for a given cfg)."""
    key = f"{master}|{cfg.encoding}|{cfg.dim}|{cfg.levels}|{cfg.quant_bits}"
    return zlib.crc32(key.encode()) & 0x7FFFFFFF

_cfg_seed = cfg_seed  # backwards-compatible alias

class MicroHDOptimizer:
    def __init__(
        self,
        baseline_config: HDCConfig,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        *,
        threshold: float = 0.01,
        epochs: int = 30,
        lr: float = 1.0,
        seed: int = 0,
        ladders: dict[str, list[int]] | None = None,
        device: str | None = None,
        verbose: bool = False,
        accept_fn: Callable[[HDCConfig, float, float], bool] | None = None,
        eval_cache: dict | None = None,
    ) -> None:
        if baseline_config.encoding not in DEFAULT_LADDERS:
            raise ValueError(baseline_config.encoding)
        ladders = ladders or DEFAULT_LADDERS[baseline_config.encoding]
        for name, values in ladders.items():
            if values[-1] != getattr(baseline_config, name):
                raise ValueError(
                    f"ladder for '{name}' must end at baseline value {getattr(baseline_config, name)}"
                )
        self.baseline_config = baseline_config
        self.ladders = ladders
        self.threshold = threshold
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = resolve_device(device)
        self.verbose = verbose
        # Optional custom acceptance rule; signature (candidate_cfg, candidate_val_acc, baseline_val_acc) -> bool
        self.accept_fn = accept_fn
        # Optional shared cache {config_key: val_acc} to avoid retraining identical
        # candidate configs across optimizer runs (e.g., different thresholds).
        self.eval_cache = eval_cache if eval_cache is not None else {}

        self.x_train = x_train.detach().to(self.device, torch.float32)
        self.y_train = y_train.detach().to(self.device, torch.long)
        self.x_val = x_val.detach().to(self.device, torch.float32)
        self.y_val = y_val.detach().to(self.device, torch.long)
        self.num_features = self.x_train.shape[1]
        self.num_classes = int(max(y_train.max(), y_val.max())) + 1

    def _train_and_eval(self, cfg: HDCConfig) -> float:
        key = (cfg.encoding, cfg.dim, cfg.levels, cfg.quant_bits)
        if key in self.eval_cache:
            return self.eval_cache[key]
        enc = build_encoder(cfg, self.num_features, seed=cfg_seed(self.seed, cfg), device=self.device)
        enc.fit_feature_range(self.x_train)
        clf = HDClassifier(enc, self.num_classes, device=self.device).fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            lr=self.lr,
            seed=self.seed,
        )
        deploy_bits = cfg.quant_bits
        pred = clf.predict(self.x_val, deploy_bits=deploy_bits)
        acc = (pred == self.y_val).float().mean().item()
        self.eval_cache[key] = acc
        return acc

    def _resources(self, cfg: HDCConfig) -> tuple[float, float]:
        from .resources import compute_ops, memory_kb

        return (
            memory_kb(cfg, self.num_features, self.num_classes),
            float(compute_ops(cfg, self.num_features, self.num_classes)),
        )

    def optimize(self) -> OptResult:
        from .resources import memory_kb

        base_acc = self._train_and_eval(self.baseline_config)

        current = self.baseline_config
        cur_idx = {name: len(vals) - 1 for name, vals in self.ladders.items()}
        lo_idx = {name: 0 for name in self.ladders}

        result = OptResult(
            config=current,
            baseline_config=self.baseline_config,
            baseline_val_acc=base_acc,
            final_val_acc=base_acc,
        )

        it = 0
        while True:
            proposals = []
            for name, ladder in self.ladders.items():
                hi = cur_idx[name]
                lo = lo_idx[name]
                if lo > hi - 1:
                    continue  # no smaller admitted value remains
                mid = (lo + hi - 1) // 2
                trial = current.with_(**{name: ladder[mid]})
                saving = (
                    memory_kb(current, self.num_features, self.num_classes)
                    - memory_kb(trial, self.num_features, self.num_classes)
                )
                proposals.append((saving, name, mid))
            if not proposals:
                break

            proposals.sort(key=lambda t: (-t[0], {"dim": 0, "levels": 1, "quant_bits": 2}.get(t[1], 3)))
            _, name, mid = proposals[0]
            ladder = self.ladders[name]
            trial = current.with_(**{name: ladder[mid]})
            acc = self._train_and_eval(trial)
            drop = base_acc - acc
            if self.accept_fn is not None:
                ok = self.accept_fn(trial, acc, base_acc)
            else:
                ok = drop <= self.threshold + 1e-12
            it += 1
            mem_before = memory_kb(current, self.num_features, self.num_classes)
            mem_after = memory_kb(trial, self.num_features, self.num_classes)
            result.steps.append(
                OptStep(it, name, getattr(current, name), getattr(trial, name), ok, acc, drop, mem_before, mem_after)
            )
            if self.verbose:
                print(
                    f"[microhd] it={it} {name}: {getattr(current, name)}->{getattr(trial, name)} "
                    f"val_acc={acc:.4f} drop={drop:+.4f} {'accept' if ok else 'reject'}"
                )
            if ok:
                current = trial
                cur_idx[name] = mid
                lo_idx[name] = 0
            else:
                lo_idx[name] = mid + 1

        result.config = current
        result.final_val_acc = self._train_and_eval(current)
        return result

def train_model(cfg: HDCConfig, num_features: int, num_classes: int, x, y, *, epochs=30, lr=1.0, seed=0, device=None):
    """Convenience helper: build encoder + classifier and fit."""
    dev = resolve_device(device)
    enc = build_encoder(cfg, num_features, seed=cfg_seed(seed, cfg), device=dev)
    enc.fit_feature_range(x)
    clf = HDClassifier(enc, num_classes, device=dev).fit(x, y, epochs=epochs, lr=lr, seed=seed)
    return clf
