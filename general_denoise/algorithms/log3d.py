from __future__ import annotations
"""Laplacian of Gaussian (3D).

- If `enhance=True`, returns `x - LoG(x)` (unsharp-like).
- Otherwise returns `LoG(x)` (edge emphasis).
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

def _log(x, sigma):
    if _cndi is not None and isinstance(x, xp.ndarray):
        return _cndi.gaussian_laplace(x, sigma=sigma, mode="nearest")
    if _ndi is None:
        raise ImportError("scipy.ndimage required for CPU gaussian_laplace.")
    xx = x.get() if hasattr(x,"get") else x
    return xp.asarray(_ndi.gaussian_laplace(xx, sigma=sigma, mode="nearest"), dtype=xp.float32)

def apply(vol, sigma=(1.0,1.0,1.0), enhance=True, iterations=1, **kwargs):
    sig = _triplef(sigma)
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        lg = _log(x, sigma=sig)
        x = x - lg if bool(enhance) else lg
    return x
