
from __future__ import annotations
from ..common.backend import gaussian_filter

def _triple(sig):
    return (sig, sig, sig) if isinstance(sig, (int, float)) else tuple(sig)

def apply(vol, sigma=(1.0, 1.0, 1.0), iterations=1, **kwargs):
    sigma = _triple(sigma)
    out = vol
    for _ in range(int(iterations)):
        out = gaussian_filter(out, sigma=sigma)
    return out
