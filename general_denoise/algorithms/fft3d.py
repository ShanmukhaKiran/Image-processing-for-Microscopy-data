from __future__ import annotations
"""
FFT-domain filtering (2D/3D).

- Linear mask M(f). Repeating the same filter `iterations` times is equivalent to a single pass with `M**iterations`.
- In 2D mode, spacing in Z is ignored for cutoff computations.
- Handles low/high/band pass with Ideal / Butterworth / Gaussian masks.

All processing in working float32; input expected in [0,1] scale.
"""
from ..common.backend import xp, as_xp

def _grids_3d(Z, Y, X, spacing=(1.0,1.0,1.0)):
    dz, dy, dx = float(spacing[0]), float(spacing[1]), float(spacing[2])
    fz = xp.fft.fftfreq(Z, d=dz) if Z>1 else xp.array([0.0], dtype=xp.float32)
    fy = xp.fft.fftfreq(Y, d=dy)
    fx = xp.fft.fftfreq(X, d=dx)
    FZ, FY, FX = xp.meshgrid(fz, fy, fx, indexing="ij")
    return xp.sqrt(FZ*FZ + FY*FY + FX*FX)

def _grids_2d(Y, X, spacing=(1.0,1.0,1.0)):
    dy, dx = float(spacing[1]), float(spacing[2])
    fy = xp.fft.fftfreq(Y, d=dy)
    fx = xp.fft.fftfreq(X, d=dx)
    FY, FX = xp.meshgrid(fy, fx, indexing="ij")
    return xp.sqrt(FY*FY + FX*FX)

def _nyq_from_spacing(spacing, mode_3d, Z, Y, X):
    if mode_3d:
        dz, dy, dx = spacing
        ny = min( (0.5/float(dz) if Z>1 else xp.inf), 0.5/float(dy), 0.5/float(dx) )
    else:
        dy, dx = spacing[1], spacing[2]
        ny = min(0.5/float(dy), 0.5/float(dx))
    return float(ny)

def _mk_mask(R, kind="butter", pass_type="low", order=2, f_lo=None, f_hi=None):
    eps = 1e-12
    kind = str(kind).lower()
    pass_type = str(pass_type).lower()
    if kind == "ideal":
        if pass_type == "low":
            M = (R <= float(f_hi)).astype(xp.float32)
        elif pass_type == "high":
            M = (R >= float(f_lo)).astype(xp.float32)
        else:
            M = xp.logical_and(R >= float(f_lo), R <= float(f_hi)).astype(xp.float32)
    elif kind == "gauss":
        if pass_type == "low":
            M = xp.exp(- (R / (float(f_hi)+eps))**2 )
        elif pass_type == "high":
            M = 1.0 - xp.exp(- (R / (float(f_lo)+eps))**2 )
        else:
            M = xp.exp(- (R/(float(f_hi)+eps))**2 ) * (1.0 - xp.exp(- (R/(float(f_lo)+eps))**2 ))
    else:  # butter
        n = int(order)
        if pass_type == "low":
            M = 1.0 / (1.0 + (R/(float(f_hi)+eps))**(2*n))
        elif pass_type == "high":
            M = 1.0 - 1.0 / (1.0 + (R/(float(f_lo)+eps))**(2*n))
        else:
            M_low  = 1.0 / (1.0 + (R/(float(f_hi)+eps))**(2*n))
            M_high = 1.0 - 1.0 / (1.0 + (R/(float(f_lo)+eps))**(2*n))
            M = M_low * M_high
    return M.astype(xp.float32, copy=False)

def apply(vol, mode="3d", filter_kind="butter", pass_type="low",
          order=2, spacing=(1.0,1.0,1.0), use_nyquist=True,
          f_lo_nyq=None, f_hi_nyq=0.25, f_lo_abs=None, f_hi_abs=None,
          shift=False, iterations=1, **kwargs):
    """
    Parameters
    ----------
    mode : {"3d","2d"}
    filter_kind : {"ideal","butter","gauss"}
    pass_type : {"low","high","band"}
    order : int
        Butterworth order (ignored for gauss/ideal).
    spacing : (dz,dy,dx)
    use_nyquist : bool
        If True, interpret f_* as fractions of min-Nyquist; else absolute cycles/unit.
    f_lo_nyq, f_hi_nyq, f_lo_abs, f_hi_abs : float or None
    shift : bool
        If True, fftshift the mask (mostly for debugging).
    iterations : int
        Repeat the same linear filter; implemented as mask**iterations in one FFT.
    """
    x = xp.asarray(vol, dtype=xp.float32)
    Z, Y, X = x.shape
    mode3d = (str(mode).lower() == "3d")

    # Use appropriate spacing for 2D mode (ignore dz)
    spacing_used = spacing if mode3d else (1.0, float(spacing[1]), float(spacing[2]))

    # Frequencies grid
    R = _grids_3d(Z,Y,X,spacing_used) if mode3d else _grids_2d(Y,X,spacing_used)
    if not mode3d:  # broadcast to Z
        R = R[None, ...] if Z>1 else R

    # Determine cutoffs
    if use_nyquist:
        ny = _nyq_from_spacing(spacing_used, mode3d, Z, Y, X)
        f_lo = (None if f_lo_nyq is None else float(f_lo_nyq) * ny)
        f_hi = (None if f_hi_nyq is None else float(f_hi_nyq) * ny)
    else:
        f_lo = (None if f_lo_abs is None else float(f_lo_abs))
        f_hi = (None if f_hi_abs is None else float(f_hi_abs))

    # Fill in defaults by pass_type
    pt = str(pass_type).lower()
    if pt == "low":
        f_hi =  f_hi if f_hi is not None else 0.25*_nyq_from_spacing(spacing_used, mode3d, Z,Y,X)
    elif pt == "high":
        f_lo =  f_lo if f_lo is not None else 0.05*_nyq_from_spacing(spacing_used, mode3d, Z,Y,X)
    else:
        f_lo =  f_lo if f_lo is not None else 0.02*_nyq_from_spacing(spacing_used, mode3d, Z,Y,X)
        f_hi =  f_hi if f_hi is not None else 0.25*_nyq_from_spacing(spacing_used, mode3d, Z,Y,X)

    # Build mask once, then raise to power=iterations to emulate repeated passes
    M = _mk_mask(R, kind=filter_kind, pass_type=pt, order=order, f_lo=f_lo, f_hi=f_hi)
    if bool(shift):
        M = xp.fft.fftshift(M, axes=(0,1,2)) if mode3d else xp.fft.fftshift(M, axes=(1,2) if Z>1 else (0,1))
    power = max(1, int(iterations))
    if power > 1:
        M = xp.power(M, power).astype(xp.float32, copy=False)

    # One FFT
    if mode3d:
        F = xp.fft.fftn(x)
        Y = xp.fft.ifftn(F * M).real
    else:
        # per-slice 2D FFT
        if Z == 1:
            F = xp.fft.fftn(x[0])
            Y0 = xp.fft.ifftn(F * M).real
            Y = Y0[None, ...]
        else:
            # broadcast M to (Z,Y,X)
            Mz = M if M.shape[0] == Z else xp.broadcast_to(M, (Z,)+M.shape[-2:])
            F = xp.fft.fftn(x, axes=(1,2))
            Y = xp.fft.ifftn(F * Mz, axes=(1,2)).real

    return Y.astype(xp.float32, copy=False)
