#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from pathlib import Path

from tqdm import tqdm
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

# Paths.
ROOT_PATH = Path(__file__).parent.resolve()


# ------------------------------------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------------------------------------

def compute_grad(U):
    grad = np.stack(np.gradient(U), axis=3)
    gradn = np.linalg.norm(grad, axis=3)
    idx = gradn > 0
    denom = 1. / gradn[idx]
    grad_normalized = grad.copy()
    for i in range(3):
        grad_normalized[..., i][idx] *= denom
    return grad_normalized


def compute_surface_pos(mask, val):
    # Find the white matter voxels.
    print("Getting the white matter surface...")
    iwm, jwm, kwm = np.nonzero(mask == val)
    # Get the positions of those voxels.
    pos = np.c_[iwm, jwm, kwm]
    return pos


def integrate_step(pos, step, grad_normalized, xyz):
    assert pos.ndim == 2
    assert pos.shape[1] == 3
    assert grad_normalized.shape == (n, m, p, 3)

    g = interpn(xyz, grad_normalized, pos)
    return pos + step * g


def integrate_field(pos, step, grad_normalized, mask, max_iter=100):
    assert pos.ndim == 2
    n_paths = pos.shape[0]
    assert pos.shape[1] == 3

    n, m, p = mask.shape
    x = np.arange(n)
    y = np.arange(m)
    z = np.arange(p)
    xyz = (x, y, z)

    out = np.zeros((n_paths, max_iter, 3), dtype=np.float32)
    out[:, 0, :] = pos
    pos_grid = np.zeros((n_paths, 3), dtype=np.int32)

    # Which positions are still in the volume and need to be integrated?
    kept = slice(None, None, None)

    for i in tqdm(range(1, max_iter)):
        out[kept, i, :] = integrate_step(
            out[kept, i - 1, :], step, grad_normalized, xyz)
        pos_grid[:] = np.round(out[:, i, :]).astype(np.int32)
        i, j, k = pos_grid.T
        kept = mask[i, j, k] != 0
        assert kept.shape == (n_paths,)
        if kept.sum() == 0:
            break

    return out


def compute_streamlines(U, mask, val):

    # Compute the normalized gradient.
    print("Computing the gradient...")
    grad_path = ROOT_PATH / 'grad.npy'
    if not grad_path.exists():
        grad_normalized = compute_grad(U)
        np.save(grad_path, grad_normalized)
    else:
        grad_normalized = np.load(grad_path, mmap_mode='r')
    assert grad_normalized.ndim == 4

    # Compute the positions of the voxels in the surface defined by mask == val.
    pos = compute_surface_pos(mask, val)

    # Integrate the gradient field from those positions.
    print("Integrating the gradient field...")
    paths = integrate_field(pos[::100000], 1, grad_normalized, mask)

    return paths


def discretize_paths(paths):
    # n_paths, path_len, _ = paths.shape
    # # paths_grid = np.round(paths).astype(np.int32)

    # i = paths_grid[..., 0]
    # j = paths_grid[..., 1]
    # k = paths_grid[..., 2]

    # streamlines = np.dstack((i, j, k))
    # return streamlines
    return np.round(paths).astype(np.int32)


# ------------------------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------------------------

def _mask_filename(region):
    region_dir = ROOT_PATH / f'regions/{region}'
    region_dir.mkdir(exist_ok=True, parents=True)
    return region_dir / f'{region}_mask.npy'


def load_mask_npy(region):
    path = _mask_filename(region)
    if not path.exists:
        return
    print(f"Loading {path}.")
    return np.load(path, mmap_mode='r')


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Load the Laplacian scalar field.
    U = np.load('U.npy')
    n, m, p = U.shape

    # Load or compute the streamlines.
    path = Path(ROOT_PATH / 'paths.npy')
    if not path.exists():
        # Load the mask (computed by the Laplacian streamlines.py script)
        mask = load_mask_npy('isocortex')

        # Compute the streamlines.
        paths = compute_streamlines(U, mask, 1)  # 1=WM, 2=GM, 3=volume

        # Save the paths
        np.save(path, paths)
    else:
        # Load the paths.
        paths = np.load(path)

    # Discretize the paths.
    print("Discretizing the streamlines...")
    streamlines = discretize_paths(paths)
    print(streamlines.shape)

    # Plotting streamlines.
    l = 50
    idx = 300
    q = 20
    qt = .97
    i = 45

    # Put the streamlines in the volume.
    i, j, k = np.transpose(streamlines[::l], (2, 1, 0))
    Uplot = U.copy()
    Uplot[i, j, k] = 2

    # Plotting code.
    x = Uplot[..., i, :]
    f = plt.figure(figsize=(8, 8))
    ax = f.subplots()
    imshow = ax.imshow(x, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    f.colorbar(imshow, ax=ax)

    ax_slider = plt.axes([0.2, 0.1, 0.05, 0.8])
    slider = Slider(
        ax_slider, "depth", valmin=0, valmax=m, valinit=i, valstep=1, orientation='vertical')

    @slider.on_changed
    def update(val):
        x = Uplot[..., val - 5:val + 5, :].max(axis=1)
        imshow.set_data(x)
        f.canvas.draw_idle()

    plt.show()
