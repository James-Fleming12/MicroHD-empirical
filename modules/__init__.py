"""MicroHD-empirical: HDC compression framework + generalization testbed."""

from .hdc import HDClassifier, HDCConfig, build_encoder
from .microhd import MicroHDOptimizer, OptResult, train_model

__all__ = [
    "HDClassifier",
    "HDCConfig",
    "build_encoder",
    "MicroHDOptimizer",
    "OptResult",
    "train_model",
]
