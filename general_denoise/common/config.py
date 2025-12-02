
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DatasetConfig:
    in_folder: str = "/home/askiran/data/"
    out_folder: str = "/home/askiran/data/denoised/"
    pattern: str = "*.tif"
    dtype: str = "uint16"
    out_dtype: str = "uint16"
    z_overlap: int = 2

@dataclass
class BaseFilterConfig:
    iterations: int = 1
    enabled: bool = True

@dataclass
class MedianConfig(BaseFilterConfig):
    size: Tuple[int, int, int] = (3, 3, 3)

@dataclass
class GaussianConfig(BaseFilterConfig):
    sigma: Tuple[float, float, float] = (1.0, 1.0, 1.0)

@dataclass
class LoGConfig(BaseFilterConfig):
    sigma: Tuple[float, float, float] = (1.0, 1.0, 1.0)

@dataclass
class UnsharpConfig(BaseFilterConfig):
    sigma: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    amount: float = 0.8
    threshold: float = 0.0

@dataclass
class FFTConfig(BaseFilterConfig):
    mode: str = "3d"            # "3d" or "2d"
    filter_kind: str = "butter" # "ideal"|"butter"|"gauss"
    pass_type: str = "low"      # "low"|"high"|"band"
    order: int = 2
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_nyquist: bool = True
    f_lo_nyq: float | None = None
    f_hi_nyq: float | None = 0.25
    f_lo_abs: float | None = None
    f_hi_abs: float | None = None
    shift: bool = False
