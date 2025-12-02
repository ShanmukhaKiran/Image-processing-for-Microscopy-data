
from __future__ import annotations
import re, hashlib, os, glob

# Short names for filters
SHORT = {
    "clahe3d":"clahe", "mean3d":"mean",
    "median3d":"med", "gaussian3d":"gauss", "unsharp3d":"unsh",
    "log3d":"log", "fft3d":"fft", "bilateral3d":"bil", "nlm3d":"nlm",
    "cdor3d":"cdor", "anisotropicdiff3d":"ad", "midrange3d":"mid",
    "wiener3d":"wien", "guided3d":"guid", "kalman3d":"kal", "knn3d":"knn",
}

# Abbreviations for parameter keys that we care to show
ABBR = {
    "size":"k", "sigma":"sig", "radius":"rad", "win":"win", "r":"r",
    "sigma_s":"sigs", "sigma_r":"sigr",
    "patch_radius":"pr", "search_radius":"sr", "h":"h",
    "amount":"amt", "threshold":"th",
    "k":"k", "lam":"lam", "func":"func",
    "order":"n",
    "filter_kind":"fk", "pass_type":"pt", "mode":"mode",
    "f_lo_nyq":"flo", "f_hi_nyq":"fhi", "f_lo_abs":"alo", "f_hi_abs":"ahi",
    "R":"R", "Q":"Q", "init_var":"ivar",
    "eps":"eps",
    "K":"K", "metric":"met", "weighting":"wgt", "w_sigma":"ws", "w_eps":"weps",
    "iterations":"i"
}

# Which keys to include per filter (kept concise but informative)
INCLUDE = {
    "median3d": ["size"],
    "gaussian3d": ["sigma"],
    "unsharp3d": ["sigma","amount","threshold"],
    "log3d": ["sigma"],
    "fft3d": ["mode","filter_kind","pass_type","order","f_lo_nyq","f_hi_nyq","f_lo_abs","f_hi_abs"],
    "bilateral3d": ["radius","sigma_s","sigma_r"],
    "nlm3d": ["patch_radius","search_radius","h"],
    "cdor3d": ["win","k_sigma","alpha"],
    "anisotropicdiff3d": ["iters","k","lam","func"],
    "midrange3d": ["win"],
    "wiener3d": ["win","sigma_n"],
    "guided3d": ["r","eps"],
    "kalman3d": ["R","Q","init_var"],
    "knn3d": ["radius","K","metric","weighting","w_sigma"],
}

def _fmt_single(v):
    if isinstance(v, float):
        s = f"{v:.3f}".rstrip('0').rstrip('.')
        return s if s else "0"
    return str(int(v)) if isinstance(v, int) else str(v)

def _fmt_val(v):
    if v is None: return None
    if isinstance(v, tuple):
        return "x".join(_fmt_single(x) for x in v)
    if isinstance(v, bool):
        return "1" if v else "0"
    return _fmt_single(v)

def format_step(name: str, params: dict) -> str:
    sn = SHORT.get(name, name)
    keys = INCLUDE.get(name, [])
    parts = []
    for k in keys:
        if k in params:
            fv = _fmt_val(params[k])
            if fv is not None: parts.append(f"{ABBR.get(k,k)}={fv}")
    # always include iterations if present
    if "iterations" in params:
        parts.append(f"{ABBR['iterations']}={_fmt_single(int(params['iterations']))}")
    return sn + "[" + ",".join(parts) + "]"

def format_sequence_suffix(sequence: list[tuple[str, dict]]) -> str:
    """sequence: [(name, params), ...] -> safe compact suffix string"""
    steps = [format_step(n, p) for (n,p) in sequence]
    s = "-".join(steps)
    # keep only safe chars for filenames
    s = re.sub(r"[^A-Za-z0-9_\-\.,=x\[\]]+", "", s)
    # trim overly long names, add short hash for uniqueness
    if len(s) > 180:
        h = hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]
        s = s[:170] + "~" + h
    return s

def rename_outputs_by_input_order(out_folder: str, output_glob: str, input_paths: list[str], suffix: str):
    """Rename files in out_folder to <input_stem>__<suffix>.tif in input order.
    Returns (ok: bool, message: str)"""
    outs = sorted(glob.glob(os.path.join(out_folder, output_glob)))
    if len(outs) != len(input_paths):
        return False, f"mismatch: outs={len(outs)} inputs={len(input_paths)}"
    for out_path, in_path in zip(outs, input_paths):
        base = os.path.splitext(os.path.basename(in_path))[0]
        new_name = f"{base}__{suffix}.tif"
        new_path = os.path.join(out_folder, new_name)
        # handle collision (rare if suffix reflects full sequence)
        if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(out_path):
            root, ext = os.path.splitext(new_path)
            i = 1
            while os.path.exists(f"{root}_{i}{ext}"):
                i += 1
            new_path = f"{root}_{i}{ext}"
        if os.path.abspath(out_path) != os.path.abspath(new_path):
            os.replace(out_path, new_path)
    return True, f"renamed {len(outs)} files"
