from __future__ import annotations
"""Wiener filter (local statistics, 3D).

- Operates in **[0,1] float32**.
- `sigma_n` is **noise std in [0,1]**. If None, auto-estimated from local variance.

Formula
-------
û = μ + max(σ²-σ_n²,0) / max(σ²,ε) * (x-μ)
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
    if isinstance(v,(list,tuple)): t=(int(v[0]),int(v[1]),int(v[2]))
    else: v=int(v); t=(v,v,v)
    return tuple([(k|1) for k in t])

def apply(vol, win=(3,5,5), sigma_n=None, iterations=1, **kwargs):
    k = _triple(win)
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        if _cndi is not None:
            m  = _cndi.uniform_filter(x, size=k)
            m2 = _cndi.uniform_filter(x*x, size=k)
        else:
            if _ndi is None: raise ImportError("scipy.ndimage required on CPU.")
            m  = xp.asarray(_ndi.uniform_filter(x.get() if hasattr(x,"get") else x, size=k))
            m2 = xp.asarray(_ndi.uniform_filter((x.get() if hasattr(x,"get") else x)**2, size=k))
        var = xp.maximum(m2 - m*m, 0.0).astype(xp.float32, copy=False)
        if sigma_n is None:
            # robust "low-side" estimate from variance map
            sn2 = xp.median(var)  # variance
        else:
            sn2 = float(sigma_n)*float(sigma_n)
        g = xp.maximum(var - sn2, 0.0) / xp.maximum(var, 1e-8)
        x = m + g * (x - m)
    return x
