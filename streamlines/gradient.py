#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import math
from pathlib import Path
import urllib.request
import shutil

from tqdm import tqdm
import numpy as np
import pywavefront
from scipy.interpolate import interpn
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from datoviz import canvas, run, colormap


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

# Paths.
ROOT_PATH = Path(__file__).parent.resolve()
ISOCORTEX = 315
N, M, P = 1320, 800, 1140
MAX_PATHS = 100_000
MAX_ITER = 1000
PATH_LEN = 100


# ------------------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------------------

def last_nonzero(arr, axis, invalid_val=-1):
    # https://stackoverflow.com/a/47269413/1595060
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def i2x(vol):
    """From volume indices to coordinates in microns (Allen CCF)."""
    return vol * 10


def x2i(vol):
    """From coordinates in microns (Allen CCF) to volume indices."""
    return np.clip(np.round(vol / 10.0).astype(np.int32), [0, 0, 0], [N-1, M-1, P-1])


def subset(paths, max_paths):
    n = paths.shape[0]
    return np.array(paths[::int(math.ceil(n / float(max_paths))), ...][:max_paths])


# ------------------------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------------------------

def region_dir(region):
    region_dir = ROOT_PATH / f'regions/{region}'
    region_dir.mkdir(exist_ok=True, parents=True)
    return region_dir


def filepath(region, fn):
    return region_dir(region) / (fn + '.npy')


def load_npy(path):
    if not path.exists():
        return
    print(f"Loading `{path}`.")
    return np.load(path, mmap_mode='r')


def save_npy(region, name, arr):
    path = filepath(region, name)
    print(f"Saving `{path}`.")
    np.save(path, arr)


def get_mesh(region_id, region):
    path = filepath(region, 'mesh')
    mesh = load_npy(path)
    if mesh is not None:
        return mesh

    # Download and save the OBJ file.
    url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/{region_id:d}.obj"
    obj_fn = region_dir(region) / f'{region}.obj'
    with urllib.request.urlopen(url) as response, open(obj_fn, 'wb') as f:
        shutil.copyfileobj(response, f)

    # Convert the OBJ to npy.
    scene = pywavefront.Wavefront(
        obj_fn, create_materials=True, collect_faces=False)
    vertices = np.array(scene.vertices, dtype=np.float32)
    np.save(path, vertices)
    return vertices


def get_mask(region):
    path = filepath(region, 'mask')
    return load_npy(path)


# ------------------------------------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------------------------------------

def compute_grad(U):
    assert U.shape == (N, M, P)
    grad = np.stack(np.gradient(U), axis=3)
    gradn = np.linalg.norm(grad, axis=3)
    idx = gradn > 0
    denom = 1. / gradn[idx]
    grad_normalized = grad.copy()
    for i in range(3):
        grad_normalized[..., i][idx] *= denom
    return grad_normalized


def get_gradient(region):
    path = filepath(region, 'gradient')
    gradient = load_npy(path)
    if gradient is not None:
        return gradient
    U = load_npy(filepath(region, 'laplacian'))
    if U is None:
        # TODO: compute the laplacian with code in streamlines.py
        raise NotImplementedError()
    assert U.ndim == 3
    gradient = compute_grad(U)
    assert gradient.ndim == 4
    np.save(path, gradient)
    del gradient
    return load_npy(path)


def integrate_step(pos, step, gradient, xyz):
    assert pos.ndim == 2
    assert pos.shape[1] == 3
    assert gradient.shape == (N, M, P, 3)
    g = interpn(xyz, gradient, pos)
    return pos - step * g


def integrate_field(pos, step, gradient, mask, max_iter=MAX_ITER, res_um=10):
    assert pos.ndim == 2
    n_paths = pos.shape[0]
    assert pos.shape == (n_paths, 3)

    n, m, p = mask.shape
    x = np.linspace(0, res_um * n, n)
    y = np.linspace(0, res_um * m, m)
    z = np.linspace(0, res_um * p, p)
    xyz = (x, y, z)

    out = np.zeros((n_paths, max_iter, 3), dtype=np.float32)
    out[:, 0, :] = pos
    pos_grid = np.zeros((n_paths, 3), dtype=np.int32)

    # Which positions are still in the volume and need to be integrated?
    kept = slice(None, None, None)

    for iter in tqdm(range(1, max_iter), desc="Integrating..."):
        prev = out[kept, iter - 1, :]
        out[kept, iter, :] = integrate_step(prev, step, gradient, xyz)
        pos_grid[:] = x2i(out[:, iter, :])
        i, j, k = pos_grid.T

        # # Get the closest voxels to find the mask.
        # kept = mask[i, j, k] != 0
        # assert kept.shape == (n_paths,)
        # if kept.sum() == 0:
        #     break

    return out


def path_lengths(paths):
    streamlines = x2i(paths)
    n_paths, path_len, _ = streamlines.shape
    d = np.abs(np.diff(paths, axis=1)).max(axis=2)
    ln = last_nonzero(d, 1)
    assert ln.shape == (n_paths,)
    return ln


def resample_paths(paths, num=PATH_LEN):
    n_paths, path_len, _ = paths.shape
    xp = np.linspace(0, 1, num)
    lengths = path_lengths(paths)
    out = np.zeros((n_paths, num, 3), dtype=np.float32)
    for i in tqdm(range(n_paths), desc="Resampling..."):
        n = lengths[i]
        if n >= 2:
            lin = interp1d(np.linspace(0, 1, n), paths[i, :n, :], axis=0)
            out[i, :num, :] = lin(xp)
        else:
            out[i, :num, :] = paths[i, 0, :]
    return out


def compute_streamlines(region, region_id):

    # Load the region mask.
    mask = get_mask(region)
    assert mask.ndim == 3

    # Download or load the mesh (initial positions of the streamlines).
    mesh = get_mesh(region_id, region)
    assert mesh.ndim == 2
    assert mesh.shape[1] == 3

    # Compute or load the gradient.
    gradient = get_gradient(region)
    assert gradient.ndim == 4
    assert gradient.shape[3] == 3

    # Integrate the gradient field from those positions.
    # Step: 10 microns
    paths = integrate_field(mesh, 1.0, gradient, mask, max_iter=MAX_ITER)

    # Resample the paths.
    streamlines = resample_paths(paths, num=PATH_LEN)

    # Save the streamlines.
    save_npy(region, 'streamlines_ibl', streamlines)


# ------------------------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------------------------

def plot_panel(panel, paths):
    assert paths.ndim == 3
    n, l, _ = paths.shape
    assert _ == 3
    length = l * np.ones(n)  # length of each path

    color = np.tile(np.linspace(0, 1, l), n)
    color = colormap(color, vmin=0, vmax=1, cmap='viridis', alpha=.75)

    v = panel.visual('line_strip', depth_test=True)
    v.data('pos', paths.reshape((-1, 3)))
    v.data('length', length)
    v.data('color', color)

    # # Points.
    # vp = panel.visual('point', depth_test=True)
    # vp.data('pos', paths[:, 0, :])
    # vp.data('color', np.array([[255, 0, 0, 128]]))


def plot_streamlines(region, max_paths=MAX_PATHS):

    paths_allen = load_npy(filepath(region, 'streamlines_allen'))
    paths_ibl = load_npy(filepath(region, 'streamlines_ibl'))

    # Subset the paths.
    paths_allen = subset(paths_allen, max_paths)
    paths_ibl = subset(paths_ibl, max_paths)

    # NOTE: convert from indices to microns
    paths_allen = (paths_allen * 10.0).astype(np.float32)

    c = canvas(show_fps=True)
    s = c.scene(cols=2)
    p0 = s.panel(col=0, controller='arcball')
    p1 = s.panel(col=1, controller='arcball')
    p0.link_to(p1)

    plot_panel(p0, paths_allen)
    plot_panel(p1, paths_ibl)

    run()


if __name__ == '__main__':
    region = 'isocortex'
    region_id = 315

    compute_streamlines(region, region_id)
    # plot_streamlines(region, MAX_PATHS)
