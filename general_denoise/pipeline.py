
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Callable, Optional
import time, gc, dataclasses
import numpy as np
from .common import io, config as C, backend as B

# Algorithms
from .algorithms import (
    median3d, gaussian3d, log3d, unsharp3d,
    fft3d, bilateral3d, nlm3d, cdor3d, anisotropicdiff3d,
    midrange3d, wiener3d, guided3d, kalman3d, knn3d
)

ALGO_REGISTRY = {
    "median3d": median3d.apply,
    "gaussian3d": gaussian3d.apply,
    "log3d": log3d.apply,
    "unsharp3d": unsharp3d.apply,
    "fft3d": fft3d.apply,
    "bilateral3d": bilateral3d.apply,
    "nlm3d": nlm3d.apply,
    "cdor3d": cdor3d.apply,
    "anisotropicdiff3d": anisotropicdiff3d.apply,
    "midrange3d": midrange3d.apply,
    "wiener3d": wiener3d.apply,
    "guided3d": guided3d.apply,
    "kalman3d": kalman3d.apply,
    "knn3d": knn3d.apply,
}

# ------------------------------
# Working-space conversion (one-time at boundary)
# ------------------------------
def _dtype_is_int(dtype_str: str) -> bool:
    s = str(dtype_str)
    return any(k in s for k in ["int8","uint8","int16","uint16","int32","uint32","int64","uint64"])

def _int_max(dtype_str: str) -> int:
    return int(np.iinfo(np.dtype(dtype_str)).max)

def to_working(host_slab: np.ndarray, in_dtype: str):
    x = B.as_xp(host_slab)
    if _dtype_is_int(in_dtype):
        x = x.astype(B.xp.float32) / max(float(_int_max(in_dtype)), 1.0)
    else:
        x = x.astype(B.xp.float32)
    return x

def from_working(work_arr, out_dtype: str):
    y = B.to_numpy(work_arr)
    if _dtype_is_int(out_dtype):
        scale = float(_int_max(out_dtype))
        y = np.clip(y, 0.0, 1.0) * scale
        y = np.rint(y).astype(out_dtype, copy=False)
    else:
        y = y.astype(out_dtype, copy=False)
    return y

# ------------------------------
# Memory model & slab sizing
# ------------------------------
_PEAK_MULT = {
    "median3d": 2.0, "gaussian3d": 2.0, "unsharp3d": 3.5, "log3d": 2.0,
    "fft3d": 8.0, "bilateral3d": 4.0, "nlm3d": 6.0, "cdor3d": 3.0,
    "anisotropicdiff3d": 4.0, "midrange3d": 3.0, "wiener3d": 4.0,
    "guided3d": 6.0, "kalman3d": 3.0, "knn3d": 8.0,
}

def _peak_mult_for_sequence(sequence: List[Tuple[str, Any]]) -> float:
    names = [name for name, _ in sequence]
    return 1.0 + max((_PEAK_MULT.get(n, 3.0) for n in names), default=2.0)

def estimate_slab_z_vram(H: int, W: int, sequence: List[Tuple[str, Any]],
                         mem_frac: float = 0.6, min_slab_z: int = 1, overlap: int = 0) -> int:
    free_bytes, _ = B.gpu_mem_info()
    bpi = B.bytes_per_item("float32")
    peak_mult = _peak_mult_for_sequence(sequence)
    if free_bytes is None:
        est = max(min_slab_z, 8)
    else:
        budget = int(free_bytes * float(mem_frac))
        voxels_budget = budget // int(bpi * peak_mult)
        est = max(min_slab_z, int(voxels_budget // (H * W)))
    return max(est, 2*overlap + 1)

# ------------------------------
# Config handling
# ------------------------------
def _as_params(obj: Any) -> Dict[str, Any]:
    """Accept dict or dataclass or objects with .to_dict()."""
    if isinstance(obj, dict):
        return dict(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    # last resort: use __dict__ but drop private attrs
    if hasattr(obj, "__dict__"):
        return {k:v for k,v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported step params type: {type(obj)}")

# ------------------------------
# Sequence application (xp arrays in, xp arrays out)
# ------------------------------
def apply_sequence(vol_working, sequence: List[Tuple[str, Any]]):
    out = vol_working
    for name, cfg in sequence:
        params = _as_params(cfg)
        if not params.pop("enabled", True):
            continue
        iters = int(params.pop("iterations", 1))
        func = ALGO_REGISTRY.get(name)
        if func is None:
            raise ValueError(f"Unknown filter: {name}")
        out = func(out, iterations=iters, **params)
    return out

def process_slab(host_slab: np.ndarray, in_dtype: str, out_dtype: str, sequence: List[Tuple[str, Any]]):
    vol = to_working(host_slab, in_dtype)
    out_work = apply_sequence(vol, sequence)
    return from_working(out_work, out_dtype)

# ------------------------------
# Folder runner (gap-free slabs, OOM auto-retry)
# ------------------------------
def run_on_folder(
    dconf: C.DatasetConfig,
    sequence: List[Tuple[str, Any]],
    slab_z: Optional[int] = None,
    overwrite: bool = True,
    progress: Optional[Callable[[int, int, str], None]] = None,
    compression: Optional[str] = None,
    mem_frac: float = 0.6,
    min_slab_z: int = 1
):
    paths = io.list_slices(dconf.in_folder, dconf.pattern)
    first = io.read_slab(paths, 0, 1, dtype=dconf.dtype)
    H, W = first.shape[-2:]
    Z_total = len(paths)
    overlap = int(dconf.z_overlap)

    if slab_z is None:
        slab_z = estimate_slab_z_vram(H, W, sequence, mem_frac=mem_frac, min_slab_z=min_slab_z, overlap=overlap)
    slab_z = max(slab_z, 2*overlap + 1)

    z_step = slab_z if overlap == 0 else max(1, slab_z - 2*overlap)

    written = 0
    start_time = time.time()
    if progress:
        progress(0, Z_total, f"Start | slab_z={slab_z} | z_step={z_step} | overlap={overlap}")

    z0 = 0
    while z0 < Z_total:
        z1 = min(Z_total, z0 + slab_z)
        try:
            host = io.read_slab(paths, z0, z1, dtype=dconf.dtype)
            host_out = process_slab(host, dconf.dtype, dconf.out_dtype, sequence)

            is_first = (z0 == 0)
            is_last  = (z1 == Z_total)
            trim_top = 0 if is_first else overlap
            trim_bot = 0 if is_last  else overlap

            if trim_top or trim_bot:
                beg = trim_top
                end = host_out.shape[0] - trim_bot
                if end <= beg:
                    raise MemoryError("Effective slab empty after trims; increase slab_z or reduce overlap.")
                host_out = host_out[beg:end]

            io.write_slab_as_folder(dconf.out_folder, host_out, z_from=(z0 + trim_top), compression=compression)

            written += int(host_out.shape[0])
            if progress:
                elapsed = time.time() - start_time
                rate = written / max(elapsed, 1e-6)
                remaining = Z_total - written
                eta_sec = remaining / max(rate, 1e-6)
                progress(written, Z_total, f"wrote z[{z0+trim_top}:{z0+trim_top+host_out.shape[0]}) | {written}/{Z_total} | ~{eta_sec:.0f}s ETA")

            z0 = z0 + z_step
            B.free_cache(); gc.collect()

        except Exception as e:
            emsg = repr(e)
            if ("OutOfMemoryError" in emsg) or ("CUDA_ERROR_OUT_OF_MEMORY" in emsg) or ("cupy.cuda.memory" in emsg) or ("Effective slab empty" in emsg):
                old = slab_z
                slab_z = max(2*overlap + 1, max(1, slab_z // 2))
                z_step = slab_z if overlap == 0 else max(1, slab_z - 2*overlap)
                if progress:
                    progress(written, Z_total, f"OOM/trim → reduce slab_z {old}→{slab_z}; new z_step={z_step}; retry")
                B.free_cache(); gc.collect()
                continue
            else:
                raise


# --- auto-register extras (added by setup cell) ---
try:
    from .algorithms import clahe3d as _clahe3d
    ALGO_REGISTRY["clahe3d"] = _clahe3d.apply
except Exception:
    pass
try:
    from .algorithms import mean3d as _mean3d
    ALGO_REGISTRY["mean3d"] = _mean3d.apply
except Exception:
    pass
