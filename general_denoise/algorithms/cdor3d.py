from __future__ import annotations
"""Contrast-Dependent Outlier Removal (3D).

- Window `win=(kz,ky,kx)` **odd**.
- Operates on **[0,1] float32**.
- `sigma_mode`: "mad" or "std" for local scale.
- `contrast`: "range" | "std" | "grad" controls threshold adaptation.
- `replace`: "soft" | "median".

This is a conservative, speckle/outlier suppressor that preserves edges by raising thresholds where contrast is high.
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
    return tuple([(k|1) for k in t])  # force odd

def _local_median_std(x, size):
    if _cndi is not None:
        m = _cndi.median_filter(x, size=size, mode="nearest")
        mu = _cndi.uniform_filter(x, size=size, mode="nearest")
        mu2= _cndi.uniform_filter(x*x, size=size, mode="nearest")
    else:
        if _ndi is None: raise ImportError("scipy.ndimage required on CPU.")
        m = xp.asarray(_ndi.median_filter(x.get() if hasattr(x,"get") else x, size=size, mode="nearest"))
        mu= xp.asarray(_ndi.uniform_filter(x.get() if hasattr(x,"get") else x, size=size, mode="nearest"))
        mu2= xp.asarray(_ndi.uniform_filter((x.get() if hasattr(x,"get") else x)**2, size=size, mode="nearest"))
    sr = xp.sqrt(xp.maximum(0.0, mu2 - mu*mu))
    return m.astype(xp.float32,copy=False), sr.astype(xp.float32,copy=False)

def apply(vol, win=(3,5,5), k_sigma=3.0, alpha=0.4,
          sigma_mode="mad", contrast="range", replace="soft", iterations=1, **kwargs):
    k = _triple(win)
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        m, sr = _local_median_std(x, size=k)
        if str(sigma_mode) == "mad":
            # robust: approximate MAD via |x - median|
            mad = _local_median_std(xp.abs(x - m), size=k)[0]
            sigma_loc = 1.4826 * mad
        else:
            sigma_loc = sr

        if str(contrast) == "std":
            C = sr
        elif str(contrast) == "grad":
            gy = xp.abs(xp.roll(x, -1, 1) - x) + xp.abs(x - xp.roll(x, 1, 1))
            gx = xp.abs(xp.roll(x, -1, 2) - x) + xp.abs(x - xp.roll(x, 1, 2))
            gz = xp.abs(xp.roll(x, -1, 0) - x) + xp.abs(x - xp.roll(x, 1, 0))
            C = (gx+gy+gz)/6.0
        else:  # range
            ymin = _cndi.minimum_filter(x, size=k) if _cndi is not None else xp.asarray(_ndi.minimum_filter(x.get() if hasattr(x,"get") else x, size=k))
            ymax = _cndi.maximum_filter(x, size=k) if _cndi is not None else xp.asarray(_ndi.maximum_filter(x.get() if hasattr(x,"get") else x, size=k))
            C = (ymax - ymin)

        thr = float(k_sigma)*sigma_loc * (1.0 + float(alpha)*C + 1e-8)
        d = x - m
        if str(replace) == "median":
            x = xp.where(xp.abs(d) > thr, m, x)
        else:  # soft
            x = xp.where(xp.abs(d) > thr, m + xp.sign(d)*thr, x)
    return x
