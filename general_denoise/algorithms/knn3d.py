from __future__ import annotations
"""k-NN denoising (3D) — reference."""
from ..common.backend import xp

def _tri(v):
    if isinstance(v,(list,tuple)): return (int(v[0]),int(v[1]),int(v[2]))
    v=int(v); return (v,v,v)

def apply(vol, radius=(2,2,2), K=9, metric="l1", weighting="inverse", w_sigma=0.1, w_eps=1e-3,
          chunk=1, tile=(512,512), iterations=1, **kwargs):
    rz,ry,rx = _tri(radius); K=int(K)
    neighbors = (2*rz+1)*(2*ry+1)*(2*rx+1)-1
    if K <= 0 or K > neighbors:
        raise ValueError(f"K must be in [1, {neighbors}] for radius={radius}")
    x = xp.asarray(vol, dtype=xp.float32)
    # (Reference: unchanged implementation; add your optimized top-K here)
    return x
