from __future__ import annotations
"""CLAHE (Contrast-Limited Adaptive Histogram Equalization)

- Works in working space **[0,1] float32**.
- `z_mode="3d"`: use GPU **cuCIM** if available, else **scikit-image** on full 3D.
- `z_mode="per_slice"`: process each Z slice independently (skimage or cuCIM 2D shim).
- Optional XY tiling for huge slices (fallback only).
- No OpenCV dependency.

Parameters
----------
clip_limit : float
    CLAHE clip limit (normalized, typical 0.01–0.03).
nbins : int
    Histogram bins (256 is fine in [0,1]).
tile : (Ty,Tx)
    Tile grid per XY.
z_mode : {"3d","per_slice"}
method : {"auto","gpu","skimage"}
iterations : int
tile_xy : (hy,hx) or None
    Optional XY tiling for massive slices (skimage fallback).

Returns
-------
xp.float32, same shape as input.
"""
from ..common.backend import xp, as_xp, to_numpy

# optional deps
try:
    import cupy as _cp
except Exception:
    _cp = None

try:
    import numpy as _np
except Exception:  # should always be there
    _np = None

try:
    import cucim.skimage as _cusk
except Exception:
    _cusk = None

try:
    from skimage import exposure as _skexp
except Exception:
    _skexp = None


def _ensure_float01(a):
    x = xp.asarray(a, dtype=xp.float32)
    return xp.clip(x, 0.0, 1.0)


def _clahe_skimage(vol_np, clip_limit, nbins, kernel_size):
    if _skexp is None:
        raise ImportError("scikit-image is required for CLAHE skimage backend.")
    return _skexp.equalize_adapthist(vol_np, clip_limit=float(clip_limit),
                                     nbins=int(nbins), kernel_size=kernel_size).astype("float32", copy=False)


def _clahe_cucim(vol_gpu, clip_limit, nbins, kernel_size):
    if _cusk is None:
        raise ImportError("cuCIM is required for CLAHE gpu backend.")
    return _cusk.exposure.equalize_adapthist(vol_gpu, clip_limit=float(clip_limit),
                                             nbins=int(nbins), kernel_size=kernel_size).astype(_cp.float32, copy=False)


def apply(vol, clip_limit=0.01, nbins=256, tile=(8,8), z_mode="per_slice",
          method="auto", iterations=1, tile_xy=None, **kwargs):
    has_cucim = (_cusk is not None)
    has_sk    = (_skexp is not None)

    # normalize tile
    try:
        tile = (int(tile[0]), int(tile[1]))
    except Exception:
        t = int(tile) if tile is not None else 8
        tile = (t, t)

    m = str(method).lower().strip()
    zm = str(z_mode).lower().strip()
    if zm not in ("3d", "per_slice"):
        raise ValueError("z_mode must be '3d' or 'per_slice'")
    use_gpu = (m == "gpu") or (m == "auto" and has_cucim)
    if use_gpu and not has_cucim:
        use_gpu = False  # fallback to skimage

    x = _ensure_float01(vol)
    Z, Y, X = x.shape
    out = x

    for _ in range(int(iterations)):
        if zm == "3d":
            if use_gpu:
                # cuCIM expects CuPy
                vol_gpu = x if (hasattr(xp, "cuda") and isinstance(x, xp.ndarray)) else _cp.asarray(to_numpy(x))
                # kernel_size for 3D: (kz, ky, kx). Use kz=1 to avoid through-plane tiles here.
                y_gpu = _clahe_cucim(vol_gpu, clip_limit=clip_limit, nbins=int(nbins),
                                     kernel_size=(1, int(tile[0]), int(tile[1])))
                out = y_gpu
            else:
                if not has_sk:
                    raise ImportError("CLAHE needs cucim or scikit-image; neither available for z_mode='3d'.")
                vol_np = to_numpy(x)
                # skimage kernel_size 3D is (kz, ky, kx); use kz=1 to avoid slicing artifacts
                y_np = _clahe_skimage(vol_np, clip_limit=clip_limit, nbins=int(nbins),
                                      kernel_size=(1, int(tile[0]), int(tile[1])))
                out = as_xp(y_np)
        else:  # per_slice
            if not (has_cucim or has_sk):
                raise ImportError("CLAHE needs cucim or scikit-image; neither available.")
            slabs = []
            hy = hx = None
            if tile_xy is not None:
                try:
                    hy, hx = int(tile_xy[0]), int(tile_xy[1])
                except Exception:
                    hy = hx = None
            for z in range(Z):
                sl_np = to_numpy(out[z])  # host float32 in [0,1]
                if use_gpu:
                    sl_gpu = _cp.asarray(sl_np)
                    y_gpu = _clahe_cucim(sl_gpu[_cp.newaxis, ...], clip_limit=clip_limit, nbins=int(nbins),
                                         kernel_size=(1, int(tile[0]), int(tile[1])))
                    y_np = _cp.asnumpy(y_gpu[0])
                else:
                    if hy and hx and _np is not None:
                        tiles = []
                        for ys in range(0, Y, hy):
                            row = []
                            for xs in range(0, X, hx):
                                block = sl_np[ys:ys+hy, xs:xs+hx]
                                row.append(_clahe_skimage(block, clip_limit=clip_limit, nbins=int(nbins),
                                                          kernel_size=(int(tile[0]), int(tile[1]))))
                            row = _np.concatenate(row, axis=1)
                            tiles.append(row)
                        y_np = _np.concatenate(tiles, axis=0)
                    else:
                        y_np = _clahe_skimage(sl_np, clip_limit=clip_limit, nbins=int(nbins),
                                              kernel_size=(int(tile[0]), int(tile[1])))
                slabs.append(y_np.astype("float32", copy=False))
            out = as_xp(_np.stack(slabs, axis=0) if _np is not None else __import__("numpy").stack(slabs, axis=0))
        x = out  # feed next iteration

    return out
