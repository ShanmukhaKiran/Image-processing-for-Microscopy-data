
from __future__ import annotations
from ..common import backend as B

def _rad_to_win(r):
    r = (r,r,r) if isinstance(r,int) else tuple(r)
    return tuple(int(2*ri+1) for ri in r)

def apply(vol, r=(1,2,2), eps=1e-3, guidance=None, iterations=1, **kwargs):
    size = _rad_to_win(r)
    out = vol
    for _ in range(int(iterations)):
        I = out
        P = I if guidance is None else guidance
        mean_I  = B.uniform_filter(I, size=size)
        mean_P  = B.uniform_filter(P, size=size)
        corr_I  = B.uniform_filter(I*I, size=size)
        corr_IP = B.uniform_filter(I*P, size=size)
        var_I   = B.maximum(corr_I - mean_I*mean_I, 0.0)
        cov_IP  = corr_IP - mean_I*mean_P
        a = cov_IP / (var_I + float(eps))
        b = mean_P - a*mean_I
        mean_a = B.uniform_filter(a, size=size)
        mean_b = B.uniform_filter(b, size=size)
        out = mean_a * I + mean_b
    return out
