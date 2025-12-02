from __future__ import annotations
"""Median filter (3D).

- Works on **[0,1] float32** internally; converts at entry.
- Use `(1, ky, kx)` to force per-slice (2D) behavior.
"""
from ..common.backend import xp

try:
    import cupyx.scipy.ndimage as _cndi
except Exception:
    _cndi = None

try:
    import scipy.ndimage as _ndi
except Exception:
    _ndi = None

def _triple(v):
    if isinstance(v, (list, tuple)):
        t = (int(v[0]), int(v[1]), int(v[2])) if len(v)==3 else (int(v[0]), int(v[0]), int(v[-1]))
    else:
        t = (int(v), int(v), int(v))
    # enforce odd
    return tuple([(k|1) for k in t])

def apply(vol, size=(3,3,3), iterations=1, **kwargs):
    k = _triple(size)
    x = xp.asarray(vol, dtype=xp.float32)
    out = x
    for _ in range(int(iterations)):
        if _cndi is not None and isinstance(out, xp.ndarray) and hasattr(xp, "cuda"):
            out = _cndi.median_filter(out, size=k, mode="nearest")
        else:
            if _ndi is None:
                raise ImportError("scipy.ndimage is required for CPU median_filter.")
            out = xp.asarray(_ndi.median_filter(out.get() if hasattr(out, "get") else out, size=k, mode="nearest"),
                             dtype=xp.float32)
    return out
