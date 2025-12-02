
from __future__ import annotations
import os, glob
import numpy as np
from tifffile import imread, imwrite

_DEFAULT_PATTERN = "*.[tT][iI][fF]*"

def list_slices(folder: str, pattern: str = _DEFAULT_PATTERN) -> list[str]:
    folder = os.fspath(folder)
    patt = pattern or _DEFAULT_PATTERN
    paths = sorted(glob.glob(os.path.join(folder, patt)))
    if not paths:
        # try fallback if user passed a narrow pattern like '*.tif'
        if patt.lower() in ("*.tif", "*.tiff"):
            patt = _DEFAULT_PATTERN
            paths = sorted(glob.glob(os.path.join(folder, patt)))
    if not paths:
        # show a small sample from the folder to help debugging
        try:
            names = sorted(os.listdir(folder))[:10]
        except Exception:
            names = []
        raise FileNotFoundError(
            f"No images in '{folder}' matching '{pattern}'. "
            f"Try pattern='{_DEFAULT_PATTERN}'. "
            f"Here are up to 10 entries in the folder: {names}"
        )
    return paths

def read_slab(paths: list[str], z_from: int, z_to: int, dtype=None) -> np.ndarray:
    slab_files = paths[z_from:z_to]
    slab = [imread(p) for p in slab_files]
    arr = np.stack(slab, axis=0)
    return arr.astype(dtype) if dtype is not None else arr

def write_slab_as_folder(out_folder: str, slab: np.ndarray, z_from: int, compression=None):
    os.makedirs(out_folder, exist_ok=True)
    for i in range(slab.shape[0]):
        imwrite(os.path.join(out_folder, f"z_{z_from+i:06d}.tif"), slab[i], compression=compression)
