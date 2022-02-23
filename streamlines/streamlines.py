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

# Distance from the cortical surface and white matter surface for every point in the isocortex mask:
# paths, paths_meta = nrrd.read(CCF_PATH / 'laplacian_10.nrrd')


# ------------------------------------------------------------------------------------------------
# Loading volume mask
# ------------------------------------------------------------------------------------------------

def load_mask_nrrd(mask_nrrd, boundary_nrrd, S1_val, S2_val):
    mask, mask_meta = nrrd.read(mask_nrrd)
    boundary, boundary_meta = nrrd.read(boundary_nrrd)

    n, m, p = boundary.shape
    assert mask.shape == (n, m, p)

    # A volume with 1 = S1, 2 = S2, 3 = in volulme
    M = np.zeros((n, m, p), dtype=np.uint8)
    M[mask != 0] = 3
    M[boundary == S1_val] = 1
    M[boundary == S2_val] = 2

    return M


def _mask_filename(region):
    region_dir = ROOT_PATH / f'regions/{region}'
    region_dir.mkdir(exist_ok=True, parents=True)
    return region_dir / f'{region}_mask.npy'


def save_mask(region, M):
    path = _mask_filename(region)
    print(f"Saving {path}.")
    np.save(path, M)


def load_mask_npy(region):
    path = _mask_filename(region)
    if not path.exists:
        return
    print(f"Loading {path}.")
    return np.load(path)


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
# Loading flatmap paths
# ------------------------------------------------------------------------------------------------

def clear_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


def bounding_box(mask):
    idx = np.nonzero(mask != 0)
    bounds = np.array([(np.min(idx[i]), np.max(idx[i])) for i in range(3)])
    box = tuple(slice(bounds[i, 0]-1, bounds[i, 1]+1, None) for i in range(3))
    nc, mc, pc = bounds[:, 1] - bounds[:, 0] + 2
    return box, (nc, mc, pc)


@jit.rawkernel()
def laplace(M, Uin, Uout, nc, mc, pc):
    # The 3 arrays M, Uin, Uout have size (nc+2, mc+2, pc+2)

    # Current voxel
    i = 1 + jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    j = 1 + jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    k = 1 + jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    if (i <= nc) and (j <= mc) and (k <= pc):
        m = M[i, j, k]
        if m == 1:
            Uout[i, j, k] = 0
        elif m == 2:
            Uout[i, j, k] = 1
        elif m == 3:
            Uout[i, j, k] = 1./6 * (Uin[i-1, j, k]+Uin[i+1, j, k] +
                                    Uin[i, j-1, k]+Uin[i, j+1, k]+Uin[i, j, k-1]+Uin[i, j, k+1])


class Runner:
    def __init__(self, M):
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
        Ua[Mgpu == 1] = 1
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
        laplace[self.grid, self.block](self.M, self.Ua, self.Ub, *self.args)
        laplace[self.grid, self.block](self.M, self.Ub, self.Ua, *self.args)

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

    def save(self, U):
        np.save('U.npy', U)

    def load(self):
        return np.load('U.npy')

    def clear(self):
        del self.M
        del self.Ua
        del self.Ub
        clear_gpu_memory()


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # annotation_path = CCF_PATH / '../scripts/annotation_10.nrrd'
    # flatmap_path = CCF_PATH / 'dorsal_flatmap_paths_10.h5'
    # load_flatmap_paths(flatmap_path, annotation_path)

    region = 'isocortex'
    iter = 100

    if not (ROOT_PATH / 'U.npy').exists():
        # Compute the Laplacian.

        # Load the boundaries of the region.
        M = load_mask_npy(region)
        # Is the uint8 mask doesn't exist, recreate it from the nrrd files.
        if M is None:
            mask_nrrd = CCF_PATH / 'isocortex_mask_10.nrrd'
            boundary_nrrd = CCF_PATH / 'isocortex_boundary_10.nrrd'
            WM = 3  # white matter
            GM = 1  # gray matter
            M = load_mask_nrrd(mask_nrrd, boundary_nrrd, WM, GM)
            save_mask('isocortex', M)

        # GPU Laplacian runner
        r = Runner(M)

        # Run X iterations
        U = r.run(iter)

        # Save the result in U.npy
        r.save(U)

    U = np.load(ROOT_PATH / 'U.npy', mmap_mode='r')
    n, m, p = U.shape

    i = 45
    x = U[..., i, :]
    f = plt.figure(figsize=(8, 8))
    ax = f.subplots()
    imshow = ax.imshow(x, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    f.colorbar(imshow, ax=ax)

    ax_slider = plt.axes([0.2, 0.1, 0.05, 0.8])
    slider = Slider(
        ax_slider, "depth", valmin=0, valmax=m, valinit=i, valstep=1, orientation='vertical')

    @slider.on_changed
    def update(val):
        x = U[..., val, :]
        imshow.set_data(x)
        f.canvas.draw_idle()

    plt.show()
