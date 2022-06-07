#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import math
from pathlib import Path
import shutil
import urllib

import nrrd
import h5py
import numba
from cupyx import jit
import cupy as cp
from tqdm import tqdm
import numpy as np


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

# Paths.
ROOT_PATH = Path(__file__).parent.resolve()
MASK_NRRD_PATH = ROOT_PATH / '../ccf_2017/isocortex_mask_10.nrrd'
BOUNDARY_NRRD_PATH = ROOT_PATH / '../ccf_2017/isocortex_boundary_10.nrrd'

# Volume shape.
N, M, P = 1320, 800, 1140

# Values used in the mask file
V_OUTSIDE = 0   # voxels outside of the surfaces and brain region
V_ST = 1        # top (outer) surface
V_VOLUME = 2    # volume between the two surfaces
V_SB = 3        # bottom (inter) surface
V_SE = 4        # intermediate surfaces

# Region used.
REGION = 'isocortex'
REGION_ID = 315


# ------------------------------------------------------------------------------------------------
# Generic data loading functions
# ------------------------------------------------------------------------------------------------

def region_dir(region):
    """Return the path to the directory containing the output data files for a given brain region.
    """
    region_dir = ROOT_PATH / f'regions/{region}'
    region_dir.mkdir(exist_ok=True, parents=True)
    return region_dir


def filepath(region, fn):
    """Return the path to an output file."""
    return region_dir(region) / (fn + '.npy')


def load_npy(path):
    """Load an NPY file in memmap read mode."""
    if not path.exists():
        print(f"Error: file {path} does not exist.")
        return
    print(f"Loading `{path}`.")
    return np.load(path, mmap_mode='r')


def save_npy(path, arr):
    """Save an array to an NPY file."""
    print(f"Saving `{path}` ({arr.shape}, {arr.dtype}).")
    np.save(path, arr)


# ------------------------------------------------------------------------------------------------
# Old data loading functions
# ------------------------------------------------------------------------------------------------

def get_mesh(region_id, region):
    """NOTE: this function is not used at the moment."""

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
    import pywavefront
    scene = pywavefront.Wavefront(
        obj_fn, create_materials=True, collect_faces=False)
    vertices = np.array(scene.vertices, dtype=np.float32)
    np.save(path, vertices)
    return vertices


def load_flatmap_paths(flatmap_path, annotation_path):
    """NOTE: this function is not used at the moment."""

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
    """Generate a single mask volume from the input mask and boundary NRRD files."""

    mask, mask_meta = nrrd.read(mask_nrrd)
    boundary, boundary_meta = nrrd.read(boundary_nrrd)

    n, m, p = boundary.shape
    assert mask.shape == (n, m, p)

    mask_ibl = mask.copy()
    mask_ibl = mask_ibl.astype(np.uint8)
    idx = boundary != 0
    mask_ibl[idx] = boundary[idx]

    return mask_ibl


def get_mask(region, mask_nrrd_path=MASK_NRRD_PATH, boundary_nrrd_path=BOUNDARY_NRRD_PATH):
    """Compute (or load from the cache) the mask volume for a given region.

    The mask volume is computed from two nrrd files (mask and boundary).

    The mask values are:

    ```python
    V_OUTSIDE = 0   # voxels outside of the surfaces and brain region
    V_ST = 1        # top (outer) surface
    V_VOLUME = 2    # volume between the two surfaces
    V_SB = 3        # bottom (inter) surface
    V_SE = 4        # intermediate surfaces
    ```

    """

    path = filepath(region, 'mask')
    if path.exists():
        return load_npy(path)
    print(f"Computing mask from the original nrrd files...")
    mask = load_mask_nrrd(mask_nrrd_path, boundary_nrrd_path)
    save_npy(path, mask)
    return load_npy(path)


def get_surface_mask(region, surf_vals):
    """Return a 3D array (volume) of booleans indicating the voxels that belong to one or several
    surfaces.

    `surf_vals` is a tuple of integers, for example `(V_ST, V_SB)` (see global constants at the
    top of this file).

    """
    mask = get_mask(region)
    surface_mask = np.isin(mask, surf_vals)
    assert surface_mask.shape == mask.shape
    assert surface_mask.dtype == bool
    return surface_mask


def get_surface_indices(region, surf_vals):
    """Return the 3D indices of the voxels belonging to one or several surfaces."""

    surface_mask = get_surface_mask(region, surf_vals)
    i, j, k = np.nonzero(surface_mask)
    pos = np.c_[i, j, k]
    return pos


# ------------------------------------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    get_mask(REGION)
