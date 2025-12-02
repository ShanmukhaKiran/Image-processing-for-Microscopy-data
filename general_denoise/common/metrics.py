
from __future__ import annotations
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def estimate_sigma_mad(x: np.ndarray) -> float:
    d = np.diff(x.astype(np.float32), axis=0)
    med = np.median(np.abs(d - np.median(d)))
    return 1.4826 * med

def quality(before: np.ndarray, after: np.ndarray) -> dict:
    return {
        "psnr": psnr(before, after, data_range=after.max() - after.min()),
        "ssim": ssim(before, after, data_range=after.max() - after.min()),
    }
