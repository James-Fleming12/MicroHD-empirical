"""Resource model of HDC workloads (Table 1 in the MicroHD paper).

Memory:
    ID-level : d * (f + l + c*q)   bits
    Proj     : d * q * (f + c)     bits

Compute proxy: elementwise binding/bundling/similarity operations per sample.
    ID-level : encode f bindings + f-1 bundlings, then c similarity dots
               -> ~ d * (2f + c)
    Proj     : encode d*f MACs, then c similarity dots
               -> ~ d * (f + c)
"""

from __future__ import annotations

from .hdc import HDCConfig

def memory_bits(cfg: HDCConfig, num_features: int, num_classes: int) -> int:
    if cfg.encoding == "id":
        return cfg.dim * (num_features + cfg.levels + num_classes * cfg.quant_bits)
    return cfg.dim * cfg.quant_bits * (num_features + num_classes)

def memory_kb(cfg: HDCConfig, num_features: int, num_classes: int) -> float:
    return memory_bits(cfg, num_features, num_classes) / 8.0 / 1024.0

def compute_ops(cfg: HDCConfig, num_features: int, num_classes: int) -> int:
    if cfg.encoding == "id":
        return cfg.dim * (2 * num_features + num_classes)
    return cfg.dim * (num_features + num_classes)
