from __future__ import annotations
"""Bilateral filter (3D).

- Spatial Gaussian (`sigma_s`) × range Gaussian (`sigma_r`) in **[0,1]** units.
- This reference uses shift loops; suitable for moderate radii.

Guards
------
- Require `sigma_r > 0`, `sigma_s > 0` to avoid degenerate weights.
"""
from ..common.backend import xp, exp as xexp

def _tri(v, as_float=False):
    if isinstance(v,(list,tuple)):
        t=(v[0],v[1],v[2])
    else:
        t=(v,v,v)
    return (float(t[0]), float(t[1]), float(t[2])) if as_float else (int(t[0]), int(t[1]), int(t[2]))

def apply(vol, radius=(1,2,2), sigma_s=(1.0,2.0,2.0), sigma_r=0.1, iterations=1, **kwargs):
    rz, ry, rx = _tri(radius)
    sz, sy, sx = _tri(sigma_s, as_float=True)
    if sigma_r <= 0 or min(sz,sy,sx) <= 0:
        raise ValueError("bilateral3d: sigma_r and sigma_s components must be > 0")
    x = xp.asarray(vol, dtype=xp.float32)
    for _ in range(int(iterations)):
        num = xp.zeros_like(x); den = xp.zeros_like(x)
        for dz in range(-rz, rz+1):
            for dy in range(-ry, ry+1):
                for dx in range(-rx, rx+1):
                    nbh = xp.roll(x, (dz,dy,dx), (0,1,2))
                    ds2 = (dz/sz)**2 + (dy/sy)**2 + (dx/sx)**2
                    w_s = xexp(-0.5 * ds2)
                    w_r = xexp(-0.5 * ( (x - nbh)**2 ) / (sigma_r*sigma_r) )
                    w = w_s * w_r
                    num += w * nbh
                    den += w
        x = xp.where(den>0, num/den, x)
    return x
