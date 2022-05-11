#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from pathlib import Path

import nrrd
import h5py
from cupyx import jit
import cupy as cp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

# Paths.
ROOT_PATH = Path(__file__).parent.resolve()
CCF_PATH = Path("../ccf_2017/").resolve()

REGION = 'isocortex'
ITERATIONS = 100
REGION_ID = 315
N, M, P = 1320, 800, 1140

# Values used in the nrrd mask
V_OUTSIDE = 0
V_S1 = 1
V_VOLUME = 2
V_S2 = 3
V_Si = 4


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


def save_npy(path, arr):
    # path = filepath(region, name)
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


# ------------------------------------------------------------------------------------------------
# Loading volume mask
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


# ------------------------------------------------------------------------------------------------
# Loading flatmap paths
# ------------------------------------------------------------------------------------------------

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
# Laplacian simulation
# ------------------------------------------------------------------------------------------------

def clear_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


def bounding_box(mask):
    idx = np.nonzero(mask != 0)
    bounds = np.array([(np.min(idx[i]), np.max(idx[i])) for i in range(3)])
    box = tuple(
        slice(bounds[i, 0] - 1, bounds[i, 1] + 1, None)
        for i in range(3))
    nc, mc, pc = bounds[:, 1] - bounds[:, 0] + 2
    return box, (nc, mc, pc)


@jit.rawkernel()
def laplace(Uin, Uout, M, nc, mc, pc):
    # The 3 arrays M, Uin, Uout have size (nc, mc, pc)

    # Current voxel
    i = 1 + jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    j = 1 + jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    k = 1 + jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    if (1 <= i) and (1 <= j) and (1 <= k) and (i <= nc - 2) and (j <= mc - 2) and (k <= pc - 2):
        m = M[i, j, k]
        if m == V_VOLUME:
            Uout[i, j, k] = 1./6 * (
                Uin[i - 1, j, k] +
                Uin[i + 1, j, k] +
                Uin[i, j - 1, k] +
                Uin[i, j + 1, k] +
                Uin[i, j, k - 1] +
                Uin[i, j, k + 1])


class Runner:
    def __init__(self, M, U=None):
        n, m, p = M.shape
        self.shape = (n, m, p)
        assert M.dtype == np.uint8

        # Compute the bounding box of the mask.
        print("Computing the bounding box of the mask volume...")
        box, (nc, mc, pc) = bounding_box(M)
        assert nc > 0
        assert mc > 0
        assert pc > 0

        # Create the M matrix encoding S1, S2, and the mask.
        size = nc * mc * pc / 1024. ** 2
        print(f"Creating one array of total size {size:.2f} MB on the GPU...")
        # Transfer M from CPU to GPU.
        Mgpu = cp.asarray(M[box])

        # Create the two scalar fields.
        size = 2 * nc * mc * pc * 4 / 1024. ** 2
        print(f"Creating two arrays of total size {size:.2f} MB on the GPU...")
        Ua = cp.zeros((nc, mc, pc), dtype=np.float32)

        if U is not None:
            print(f"Starting from the existing laplacian array {U.shape}.")
            Ua[...] = cp.asarray(U[box])
        else:
            Ua[Mgpu == 1] = 1
            Ua[Mgpu == 2] = 2

        Ub = Ua.copy()

        # CUDA grid and block.
        b = 8
        self.block = (b, b, b)
        self.grid = (int(np.ceil(nc / float(b))),
                     int(np.ceil(mc / float(b))), int(np.ceil(pc / float(b))))

        # Main loop
        self.args = (cp.int32(nc), cp.int32(mc), cp.int32(pc))

        self.M = Mgpu
        self.Ua = Ua
        self.Ub = Ub
        self.box = box

    def iter(self):
        # ping-pong between the 2 arrays to avoid edge-effects while computing
        # the Laplacian in parallel on the GPU.
        # NOTE: each Python iteration here is actually made of 2 algorithm iterations
        laplace[self.grid, self.block](self.Ua, self.Ub, self.M, *self.args)
        laplace[self.grid, self.block](self.Ub, self.Ua, self.M, *self.args)

    def run(self, iterations):
        assert iterations > 0
        for i in tqdm(range(iterations)):
            self.iter()
            if i % 10 == 0:
                cp.cuda.stream.get_current_stream().synchronize()

        # Construct the final result.
        print("Constructing the output array...")
        Uout = np.zeros(self.shape, dtype=np.float32)
        Uout[self.box] = cp.asnumpy(self.Ua)

        return Uout

    def clear(self):
        del self.M
        del self.Ua
        del self.Ub
        clear_gpu_memory()


def compute_laplacian():

    mask = get_mask(REGION)
    assert mask.ndim == 3
    assert mask.shape == (N, M, P)

    # Load the current result.
    U = load_npy(filepath(REGION, 'laplacian'))

    # GPU Laplacian runner
    r = Runner(mask, U=U)

    # Run X iterations
    U = r.run(ITERATIONS)

    save_npy(filepath(REGION, 'laplacian'), U)


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    compute_laplacian()
    U = load_npy(filepath(REGION, 'laplacian'))

    plt.imshow(U[500, :, :], interpolation='none', origin='lower')
    plt.show()
