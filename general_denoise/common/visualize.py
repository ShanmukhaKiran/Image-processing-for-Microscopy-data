
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def show_pair(vol_a: np.ndarray, vol_b: np.ndarray, z: int | None = None, title_a="Input", title_b="Output"):
    if z is None:
        z = vol_a.shape[0] // 2
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1); ax.imshow(vol_a[z], cmap="gray"); ax.set_title(title_a); ax.axis("off")
    ax = fig.add_subplot(1, 2, 2); ax.imshow(vol_b[z], cmap="gray"); ax.set_title(title_b); ax.axis("off")
    plt.show()
