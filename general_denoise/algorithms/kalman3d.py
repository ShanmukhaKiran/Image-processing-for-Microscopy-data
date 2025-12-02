
from __future__ import annotations
from ..common import backend as B

def apply(vol, R=0.01, Q=1e-4, init_var=1.0, iterations=1, **kwargs):
    """Simple per-voxel 1D Kalman along Z.
    vol: xp array [Z, Y, X] in working space (float32 0..1 recommended)
    R: measurement noise variance
    Q: process noise variance
    init_var: initial state variance
    iterations: apply forward pass multiple times
    """
    xp = B.xp
    vol = vol  # already xp by pipeline

    # ensure xp scalars (so CuPy math doesn't see NumPy scalars)
    Rv = xp.asarray(R, dtype=xp.float32)
    Qv = xp.asarray(Q, dtype=xp.float32)
    initv = xp.asarray(init_var, dtype=xp.float32)

    out = vol
    for _ in range(int(iterations)):
        Z = out.shape[0]
        # state and variance per-pixel
        xf = out[0].astype(xp.float32, copy=True)
        Pf = xp.full_like(xf, initv, dtype=xp.float32)

        y = xp.empty_like(out, dtype=out.dtype)
        y[0] = xf

        for k in range(1, Z):
            zk = out[k]
            K = Pf / (Pf + Rv)           # Kalman gain
            xf = xf + K * (zk - xf)      # update estimate
            Pf = (1.0 - K) * Pf + Qv     # update variance
            y[k] = xf

        out = y
    return out
