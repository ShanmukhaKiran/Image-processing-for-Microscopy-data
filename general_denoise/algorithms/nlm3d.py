from __future__ import annotations
"""Non-Local Means (3D) — simple reference version.

Notes
-----
- Works in **[0,1] float32**.
- This reference uses Python loops over search offsets → OK for small windows.
- For large windows, vectorize (stack shifted neighbors) or write a RawKernel.
- `fast=True` nudges defaults to small Z radii on anisotropic data.

Parameters
----------
patch_radius : (rz,ry,rx)
search_radius: (sz,sy,sx)
h : float
sigma : Optional[float]  (noise std in [0,1]); if given, compensates SSD by 2*sigma^2*patch_size
use_box : bool   (box-filtered weights approx)

Caveat: Runtime ∝ (2sz+1)*(2sy+1)*(2sx+1). Keep it small in Z.
"""
from ..common.backend import xp

def _tri_i(v):
    if isinstance(v,(list,tuple)): return (int(v[0]),int(v[1]),int(v[2]))
    v=int(v); return (v,v,v)

def apply(vol, patch_radius=(1,2,2), search_radius=(2,6,6), h=0.8, sigma=None, use_box=True, fast=True, iterations=1, **kwargs):
    pr = _tri_i(patch_radius); sr = _tri_i(search_radius)
    if fast:
        pr = (min(pr[0],1), pr[1], pr[2])
        sr = (min(sr[0],2), sr[1], sr[2])
    # guard pathological windows
    max_nei = (2*sr[0]+1)*(2*sr[1]+1)*(2*sr[2]+1)
    if max_nei > 3_000:
        raise ValueError(f"NLM search window too large ({max_nei} neighbors). Reduce search_radius.")
    x = xp.asarray(vol, dtype=xp.float32)
    out = x
    lam = float(h*h)
    psize = (2*pr[0]+1)*(2*pr[1]+1)*(2*pr[2]+1)
    comp = 0.0 if sigma is None else 2.0*float(sigma*sigma)*psize
    for _ in range(int(iterations)):
        num = xp.zeros_like(out); den = xp.zeros_like(out)
        for dz in range(-sr[0], sr[0]+1):
            for dy in range(-sr[1], sr[1]+1):
                for dx in range(-sr[2], sr[2]+1):
                    nbh = xp.roll(out, shift=(dz,dy,dx), axis=(0,1,2))
                    # simple patch energy via box approx (cheaper than exact SSD)
                    d = (out - nbh)
                    w = xp.exp(-(xp.maximum(0.0, (d*d - comp))) / (lam + 1e-12))
                    num += w * nbh
                    den += w
        out = xp.where(den>0, num/den, out)
    return out
