#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import math
from pathlib import Path

import nrrd
import h5py
import numba
from cupyx import jit
import cupy as cp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

# Paths.
ROOT_PATH = Path(__file__).parent.resolve()
CCF_PATH = Path("../ccf_2017/").resolve()

# Values used in the nrrd mask
V_OUTSIDE = 0
V_S1 = 1
V_VOLUME = 2
V_S2 = 3
V_Si = 4

REGION = 'isocortex'
REGION_ID = 315
N, M, P = 1320, 800, 1140


# ------------------------------------------------------------------------------------------------
# Generic data loading functions
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


def save_npy(path, arr):
    # path = filepath(region, name)
    print(f"Saving `{path}` ({arr.shape}, {arr.dtype}).")
    np.save(path, arr)


# ------------------------------------------------------------------------------------------------
# Old data loading functions
# ------------------------------------------------------------------------------------------------

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


def load_flatmap_paths(flatmap_path, annotation_path):
    with h5py.File(flatmap_path, 'r+') as f:
        dorsal_paths = f['paths'][:]
        dorsal_lookup = f['view lookup'][:]

    n_lines, max_length = dorsal_paths.shape

    # Dorsal lookup:
    ap, ml = dorsal_lookup.shape  # every item is an index

    idx = (dorsal_lookup != 0)
    # All unique indices appearing in dorsal_paths:
    ids = dorsal_lookup[idx]

    # The (ap, ml) pair for each unique index:
    apml = np.c_[np.nonzero(idx)]

    i = 10000
    dpath = dorsal_paths[i]
    dpath = dpath[dpath != 0]
    ccf10, meta = nrrd.read(annotation_path)
    line = np.c_[np.unravel_index(dpath, ccf10.shape)]

    return line


# ------------------------------------------------------------------------------------------------
# Mask
# ------------------------------------------------------------------------------------------------

def load_mask_nrrd(mask_nrrd, boundary_nrrd):
    mask, mask_meta = nrrd.read(mask_nrrd)
    boundary, boundary_meta = nrrd.read(boundary_nrrd)

    n, m, p = boundary.shape
    assert mask.shape == (n, m, p)

    mask_ibl = mask.copy()
    mask_ibl = mask_ibl.astype(np.uint8)
    idx = boundary != 0
    mask_ibl[idx] = boundary[idx]

    return mask_ibl


def get_mask(region):
    path = filepath(region, 'mask')
    if path.exists():
        return load_npy(path)
    print(f"Computing mask from the original nrrd files...")
    mask_nrrd = ROOT_PATH / '../ccf_2017/isocortex_mask_10.nrrd'
    boundary_nrrd = ROOT_PATH / '../ccf_2017/isocortex_boundary_10.nrrd'
    mask = load_mask_nrrd(mask_nrrd, boundary_nrrd)
    save_npy(path, mask)
    return load_npy(path)


def get_surface_mask(region, surf_vals):
    mask = get_mask(region)
    surface_mask = np.isin(mask, surf_vals)
    assert surface_mask.shape == mask.shape
    assert surface_mask.dtype == bool
    return surface_mask


def get_surface_indices(region, surf_vals):
    surface_mask = get_surface_mask(region, surf_vals)
    i, j, k = np.nonzero(surface_mask)
    pos = np.c_[i, j, k]
    return pos


if __name__ == '__main__':
    get_mask(REGION)
