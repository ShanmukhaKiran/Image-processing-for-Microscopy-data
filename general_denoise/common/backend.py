
from __future__ import annotations
import importlib
import numpy as _np

# Try GPU stack (CuPy)
try:
    _cp = importlib.import_module("cupy")
    _cupyx = importlib.import_module("cupyx")
    _GPU = True
except Exception:
    _cp = None
    _cupyx = None
    _GPU = False

# Public array module alias
xp = _cp if _GPU else _np

# ndimage backends
_cux_nd = None
if _GPU:
    try:
        _cux_nd = importlib.import_module("cupyx.scipy.ndimage")
    except Exception:
        _cux_nd = None
from scipy import ndimage as _sp_nd

# ------------------------------------------------------------------
# Policy: strict xp arrays inside algorithms (no silent ping-pong)
# ------------------------------------------------------------------
STRICT_XP = True

def as_xp(a, dtype=None):
    """Move host (NumPy) to xp (CuPy if available). Call once at the pipeline boundary."""
    return (xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a))

def to_numpy(a):
    """Move xp array back to NumPy. No-op if already NumPy."""
    if _GPU and hasattr(a, "__cuda_array_interface__"):
        return _cp.asnumpy(a)
    return _np.asarray(a)

def _ensure_xp(a):
    """Assert `a` is already an xp array (CuPy on GPU path, NumPy on CPU path)."""
    if not STRICT_XP:
        return xp.asarray(a)
    if _GPU:
        if not isinstance(a, _cp.ndarray):
            raise TypeError("Expected CuPy array in GPU path; convert once at boundary with as_xp().")
    else:
        if not isinstance(a, _np.ndarray):
            raise TypeError("Expected NumPy array in CPU path.")
    return a

# --------------------------- filters -------------------------------
def gaussian_filter(a, sigma):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.gaussian_filter(a, sigma=sigma, mode="reflect")
    return _sp_nd.gaussian_filter(a, sigma=sigma, mode="reflect")

def median_filter(a, size):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.median_filter(a, size=size, mode="reflect")
    return _sp_nd.median_filter(a, size=size, mode="reflect")

def uniform_filter(a, size):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.uniform_filter(a, size=size, mode="reflect")
    return _sp_nd.uniform_filter(a, size=size, mode="reflect")

def minimum_filter(a, size):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.minimum_filter(a, size=size, mode="reflect")
    return _sp_nd.minimum_filter(a, size=size, mode="reflect")

def maximum_filter(a, size):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.maximum_filter(a, size=size, mode="reflect")
    return _sp_nd.maximum_filter(a, size=size, mode="reflect")

def sobel(a, axis):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.sobel(a, axis=axis)
    return _sp_nd.sobel(a, axis=axis, mode="reflect")

def gaussian_laplace(a, sigma):
    a = _ensure_xp(a)
    if _GPU and _cux_nd is not None:
        return _cux_nd.gaussian_laplace(a, sigma=sigma)
    return _sp_nd.gaussian_laplace(a, sigma=sigma, mode="reflect")

# ----------------------------- FFT ---------------------------------
def fftn(a):
    a = _ensure_xp(a)
    return (_cp.fft.fftn(a) if _GPU else _np.fft.fftn(a))

def ifftn(a):
    a = _ensure_xp(a)
    return (_cp.fft.ifftn(a) if _GPU else _np.fft.ifftn(a))

def fftfreq(n, d=1.0):
    return (_cp.fft.fftfreq(n, d=d) if _GPU else _np.fft.fftfreq(n, d=d))

# ------------------------- math helpers ----------------------------
def roll(a, shift, axis):
    a = _ensure_xp(a); return xp.roll(a, shift, axis=axis)

def _is_scalar(x):
    if isinstance(x, (int, float, bool)):
        return True
    if isinstance(x, _np.generic):
        return True
    try:
        return getattr(x, "ndim", None) == 0
    except Exception:
        return False

def _as_xp_scalar_like(x, like):
    if _is_scalar(x):
        return xp.asarray(x, dtype=getattr(like, "dtype", None))
    return x

def exp(a):
    if _is_scalar(a):
        return xp.exp(xp.asarray(a))
    a = _ensure_xp(a); return xp.exp(a)

def sqrt(a):
    if _is_scalar(a):
        return xp.sqrt(xp.asarray(a))
    a = _ensure_xp(a); return xp.sqrt(a)

def abs(a):
    a = _ensure_xp(a); return xp.abs(a)

def clip(a, lo, hi):
    a = _ensure_xp(a); return xp.clip(a, lo, hi)

def maximum(a, b):
    a = _ensure_xp(a)
    b = _as_xp_scalar_like(b, a)
    if not _is_scalar(b):
        b = _ensure_xp(b)
    return xp.maximum(a, b)

def minimum(a, b):
    a = _ensure_xp(a)
    b = _as_xp_scalar_like(b, a)
    if not _is_scalar(b):
        b = _ensure_xp(b)
    return xp.minimum(a, b)

def zeros_like(a, dtype=None):
    a = _ensure_xp(a); return xp.zeros_like(a, dtype=dtype)

def empty_like(a, dtype=None):
    a = _ensure_xp(a); return xp.empty_like(a, dtype=dtype)

# ----------------------------- misc --------------------------------
def bytes_per_item(dtype) -> int:
    return xp.dtype(dtype).itemsize

def gpu_mem_info():
    if not _GPU:
        return (None, None)
    free, total = _cp.cuda.runtime.memGetInfo()
    return int(free), int(total)

def free_cache():
    if _GPU:
        _cp.get_default_memory_pool().free_all_blocks()
        try:
            _cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass

def get_cupy():
    return _cp if _GPU else None

def is_gpu():
    return _GPU
