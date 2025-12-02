from __future__ import annotations
"""Morphological midrange filter (3D): (local min + local max)/2 in window."""
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

def apply(vol, win=(3,5,5), iterations=1, **kwargs):
    k = _triple(win)
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        if _cndi is not None:
            mn = _cndi.minimum_filter(x, size=k)
            mx = _cndi.maximum_filter(x, size=k)
        else:
            if _ndi is None: raise ImportError("scipy.ndimage required on CPU.")
            xx = x.get() if hasattr(x,"get") else x
            mn = xp.asarray(_ndi.minimum_filter(xx, size=k))
            mx = xp.asarray(_ndi.maximum_filter(xx, size=k))
        x = 0.5*(mn+mx)
    return x
