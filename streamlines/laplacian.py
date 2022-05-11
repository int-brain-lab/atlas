#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

REGION = 'isocortex'
REGION_ID = 315
N, M, P = 1320, 800, 1140
ITERATIONS = 1000


# ------------------------------------------------------------------------------------------------
# Normal
# ------------------------------------------------------------------------------------------------

def compute_normal(mask):
    i, j, k = np.nonzero(np.isin(mask, (V_S1, V_S2, V_Si)))
    vi0 = (mask[i-1,j,k] == V_VOLUME).astype(np.int8)
    vi1 = (mask[i+1,j,k] == V_VOLUME).astype(np.int8)
    vj0 = (mask[i,j-1,k] == V_VOLUME).astype(np.int8)
    vj1 = (mask[i,j+1,k] == V_VOLUME).astype(np.int8)
    vk0 = (mask[i,j,k-1] == V_VOLUME).astype(np.int8)
    vk1 = (mask[i,j,k+1] == V_VOLUME).astype(np.int8)
    count = vi0 + vi1 + vj0 + vj1 + vk0 + vk1  # (n,)
    pos = np.c_[i,j,k]
    normal = (
        np.c_[i-1,j,k] * vi0[:, np.newaxis] +
        np.c_[i+1,j,k] * vi1[:, np.newaxis] +
        np.c_[i,j-1,k] * vj0[:, np.newaxis] +
        np.c_[i,j+1,k] * vj1[:, np.newaxis] +
        np.c_[i,j,k-1] * vk0[:, np.newaxis] +
        np.c_[i,j,k+1] * vk1[:, np.newaxis] -
        count[:, np.newaxis] * pos)

    Ni = np.zeros((N, M, P), dtype=np.int8)
    Nj = np.zeros((N, M, P), dtype=np.int8)
    Nk = np.zeros((N, M, P), dtype=np.int8)

    Ni[i, j, k] = normal[:, 0]
    Nj[i, j, k] = normal[:, 1]
    Nk[i, j, k] = normal[:, 2]

    return np.stack((Ni, Nj, Nk), axis=-1)


def get_normal(region):
    path = filepath(region, 'normal')
    if path.exists():
        return load_npy(path)
    print(f"Computing surface normal...")
    normal = compute_normal(get_mask(region))
    save_npy(path, normal)
    return load_npy(path)


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


@jit.rawkernel()
def vonneumann(Uout, M, Ni, Nj, Nk, nc, mc, pc):
    # The 3 arrays M, Uin, Uout have size (nc, mc, pc)

    # Current voxel
    i = 1 + jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    j = 1 + jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    k = 1 + jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    if (1 <= i) and (1 <= j) and (1 <= k) and (i <= nc - 2) and (j <= mc - 2) and (k <= pc - 2):
        m = M[i, j, k]
        # Direction of streamlines: S1 (val=1) ==> S2 (val=3)
        if m  == V_S1 or m == V_S2 or m == V_Si: # NOTE: remove the "m == V_Si" part ??
            ni = Ni[i, j, k]
            nj = Nj[i, j, k]
            nk = Nk[i, j, k]
            # Reverse the gradient for one of the surfaces
            if m == V_S2 or m == V_Si:
                ni, nj, nk = -ni, -nj, -nk

            Uout[i, j, k] = (
                Uout[i+ni,j,k] +
                Uout[i,j+nj,k] +
                Uout[i,j,k+nk] +
                1) / 3.0  # NOTE: or -1??


class Runner:
    def __init__(self, M, normal, U=None):
        n, m, p = M.shape
        self.shape = (n, m, p)
        assert M.dtype == np.uint8
        assert normal.dtype == np.int8
        assert normal.shape == self.shape + (3,)

        # Compute the bounding box of the mask.
        print("Computing the bounding box of the mask volume...")
        box, (nc, mc, pc) = bounding_box(M)
        assert nc > 0
        assert mc > 0
        assert pc > 0

        # Mask.
        size = nc * mc * pc / 1024. ** 2
        print(f"Creating mask array of total size {size:.2f} MB on the GPU...")
        # Transfer M from CPU to GPU.
        Mgpu = cp.asarray(M[box])

        # Normal.
        size = 3 * nc * mc * pc / 1024. ** 2
        print(f"Creating 3 normal arrays of total size {size:.2f} MB on the GPU...")
        # Transfer M from CPU to GPU.
        Ni_gpu = cp.asarray(normal[..., 0][box])
        Nj_gpu = cp.asarray(normal[..., 1][box])
        Nk_gpu = cp.asarray(normal[..., 2][box])

        # Create the two scalar fields.
        size = 2 * nc * mc * pc * 4 / 1024. ** 2
        print(f"Creating two arrays of total size {size:.2f} MB on the GPU...")
        Ua = cp.zeros((nc, mc, pc), dtype=np.float32)

        if U is not None:
            print(f"Starting from the existing laplacian array {U.shape}.")
            Ua[...] = cp.asarray(U[box])
        else:
            # Initial values: the same as the mask.
            Ua[...] = Mgpu

        Ub = Ua.copy()

        # CUDA grid and block.
        b = 8
        self.block = (b, b, b)
        self.grid = (int(np.ceil(nc / float(b))),
                     int(np.ceil(mc / float(b))), int(np.ceil(pc / float(b))))

        # Main loop
        self.args = (cp.int32(nc), cp.int32(mc), cp.int32(pc))

        self.M = Mgpu
        self.Ni = Ni_gpu
        self.Nj = Nj_gpu
        self.Nk = Nk_gpu
        self.Ua = Ua
        self.Ub = Ub
        self.box = box

    def iter(self):
        # ping-pong between the 2 arrays to avoid edge-effects while computing
        # the Laplacian in parallel on the GPU.
        # NOTE: each Python iteration here is actually made of 2 algorithm iterations
        laplace[self.grid, self.block](self.Ua, self.Ub, self.M, *self.args)
        vonneumann[self.grid, self.block](self.Ub, self.M, self.Ni, self.Nj, self.Nk, *self.args)

        laplace[self.grid, self.block](self.Ub, self.Ua, self.M, *self.args)
        vonneumann[self.grid, self.block](self.Ua, self.M, self.Ni, self.Nj, self.Nk, *self.args)

    def run(self, iterations):
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
        del self.Ni_gpu
        del self.Nj_gpu
        del self.Nk_gpu
        clear_gpu_memory()


def compute_laplacian():

    mask = get_mask(REGION)
    assert mask.ndim == 3
    assert mask.shape == (N, M, P)
    # print(np.bincount(mask.ravel()))

    normal = get_normal(REGION)
    assert normal.ndim == 4
    assert normal.shape == (N, M, P, 3)

    # Load the current result.
    U = load_npy(filepath(REGION, 'laplacian'))

    # GPU Laplacian runner
    r = Runner(mask, normal, U=U)

    # Run X iterations
    U = r.run(ITERATIONS)

    save_npy(filepath(REGION, 'laplacian'), U)


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # compute_laplacian()
    U = load_npy(filepath(REGION, 'laplacian'))

    i0 = 500
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Colormap scaling.
    # Umin, Umax = U.min(), U.max()
    q = .001
    Umin = np.quantile(U.ravel(), q)
    Umax = np.quantile(U.ravel(), 1-q)
    norm = Normalize(vmin=Umin, vmax=Umax)

    ims = ax.imshow(U[i0, :, :], interpolation='none', origin='lower', norm=norm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ims, cax=cax, orientation='vertical')

    axs = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=axs,
        label="i",
        valmin=0,
        valstep=1,
        valmax=N-1,
        valinit=i0,
        orientation="horizontal"
    )

    @slider.on_changed
    def update(i):
        i = int(i)
        ims.set_data(U[i, :, :])
        fig.canvas.draw_idle()

    plt.show()
