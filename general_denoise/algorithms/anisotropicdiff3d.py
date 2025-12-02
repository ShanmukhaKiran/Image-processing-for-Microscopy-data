from __future__ import annotations
"""Perona–Malik anisotropic diffusion (3D).

- `iters` = **inner** diffusion iterations.
- `iterations` (pipeline) repeats the whole pass (**outer** loop).
- Simple 6-neighbour scheme; keep `lam ≤ 0.25`.
- All in **[0,1] float32**.

Tip: A fused CuPy RawKernel (computing divergence + update) would greatly reduce Python overhead.
"""
from ..common.backend import xp, exp as xexp

def _triplef(v):
    return (float(v[0]), float(v[1]), float(v[2])) if isinstance(v,(list,tuple)) else (float(v),)*3

def apply(vol, iters=50, k=0.05, lam=0.15, func="exp", vox=(1.0,1.0,1.0), iterations=1, **kwargs):
    vz, vy, vx = _triplef(vox)
    k = float(k); lam=float(lam); iters=int(iters)
    x = xp.asarray(vol, dtype=xp.float32)
    for _outer in range(int(iterations)):
        u = x
        for _ in range(iters):
            # forward differences
            uzp = xp.roll(u, -1, 0) - u
            uyp = xp.roll(u, -1, 1) - u
            uxp = xp.roll(u, -1, 2) - u
            # conductance
            if str(func) == "lorentz":
                cz = 1.0 / (1.0 + (uzp/(k*vx))**2)
                cy = 1.0 / (1.0 + (uyp/(k*vy))**2)
                cx = 1.0 / (1.0 + (uxp/(k*vx))**2)
            else:
                cz = xexp(-(uzp/(k*vz))**2)
                cy = xexp(-(uyp/(k*vy))**2)
                cx = xexp(-(uxp/(k*vx))**2)
            div = (cz*uzp - xp.roll(cz, 1, 0)*xp.roll(uzp, 1, 0))                 + (cy*uyp - xp.roll(cy, 1, 1)*xp.roll(uyp, 1, 1))                 + (cx*uxp - xp.roll(cx, 1, 2)*xp.roll(uxp, 1, 2))
            u = u + lam * div
        x = u
    return x
