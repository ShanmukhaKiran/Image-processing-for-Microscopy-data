from __future__ import annotations
"""Mean (box) filter, 3D.

- Baseline linear smoothing; blurs edges.
- Optional `enforce_odd=True` forces symmetric kernel sizes.
"""
from ..common.backend import xp, to_numpy, as_xp
try:
    import cupyx.scipy.ndimage as _cndi
except Exception:
    _cndi = None
try:
    import scipy.ndimage as _ndi
except Exception:
    _ndi = None

def _tri(v):
    if isinstance(v,(list,tuple)):
        if len(v)==3: t=(int(v[0]),int(v[1]),int(v[2]))
        elif len(v)==2: t=(int(v[0]),int(v[0]),int(v[1]))
        else: t=(int(v[0]),)*3
    else:
        t=(int(v),)*3
    return t

def apply(vol, size=(3,3,3), iterations=1, mode="reflect", enforce_odd=False, **kwargs):
    k = _tri(size)
    if enforce_odd:
        k = tuple([(s|1) for s in k])
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        if _cndi is not None and isinstance(x, xp.ndarray):
            x = _cndi.uniform_filter(x, size=k, mode=mode)
        else:
            if _ndi is None: raise ImportError("scipy.ndimage required for CPU uniform_filter.")
            x = as_xp(_ndi.uniform_filter(to_numpy(x), size=k, mode=mode)).astype(xp.float32, copy=False)
    return x
