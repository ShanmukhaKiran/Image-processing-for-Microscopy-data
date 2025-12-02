from __future__ import annotations
"""Unsharp masking (3D).

- Threshold is in **[0,1]** working units; avoids boosting small noise.
- Set `sigma=(0, sy, sx)` to force 2D behavior.
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

def _triplef(v):
    if isinstance(v,(list,tuple)): return (float(v[0]),float(v[1]),float(v[2]))
    v=float(v); return (v,v,v)

def _gauss(x, sigma):
    if _cndi is not None and isinstance(x, xp.ndarray):
        return _cndi.gaussian_filter(x, sigma=sigma, mode="nearest")
    if _ndi is None:
        raise ImportError("scipy.ndimage required for CPU gaussian_filter.")
    xx = x.get() if hasattr(x,"get") else x
    return xp.asarray(_ndi.gaussian_filter(xx, sigma=sigma, mode="nearest"), dtype=xp.float32)

def apply(vol, sigma=(1.0,1.0,1.0), amount=0.6, threshold=0.0, clip_min=None, clip_max=None, iterations=1, **kwargs):
    sig = _triplef(sigma)
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        blur = _gauss(x, sigma=sig)
        detail = x - blur
        if threshold and threshold>0:
            detail = xp.where(xp.abs(detail) >= float(threshold), detail, 0.0)
        x = x + float(amount)*detail
        if clip_min is not None or clip_max is not None:
            lo = -xp.inf if clip_min is None else float(clip_min)
            hi =  xp.inf if clip_max is None else float(clip_max)
            x = xp.clip(x, lo, hi)
    return x
