#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *
from surface import *


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

ITERATIONS = 5000


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
    margin = 6
    box = tuple(
        slice(bounds[i, 0] - margin, bounds[i, 1] + margin, None)
        for i in range(3))
    nc, mc, pc = bounds[:, 1] - bounds[:, 0] + 2 * margin
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
        if m == V_S1 or m == V_S2 or m == V_Si:  # NOTE: remove the "m == V_Si" part ??
            ni = Ni[i, j, k]
            nj = Nj[i, j, k]
            nk = Nk[i, j, k]
            # Reverse the gradient for one of the surfaces
            if m == V_S2 or m == V_Si:
                ni, nj, nk = -ni, -nj, -nk

            Uout[i, j, k] = (
                Uout[i+ni, j, k] +
                Uout[i, j+nj, k] +
                Uout[i, j, k+nk] +
                1) / 3.0  # NOTE: or -1??


class Runner:
    def __init__(self, mask, normal, U=None):
        n, m, p = mask.shape
        self.shape = (n, m, p)
        assert mask.dtype == np.uint8
        assert normal.dtype == np.float32
        assert normal.shape == self.shape + (3,)

        # Compute the bounding box of the mask.
        print("Computing the bounding box of the mask volume...")
        box, (nc, mc, pc) = bounding_box(mask)

        assert nc > 0
        assert mc > 0
        assert pc > 0

        # Mask.
        size = nc * mc * pc / 1024. ** 2
        print(f"Creating mask array of total size {size:.2f} MB on the GPU...")
        # Transfer the pask to the GPU.
        mask_gpu = cp.asarray(mask[box])

        # TODO: split the mask in 2 hemispheres, with padding

        # Normal.
        size = 3 * nc * mc * pc * 4 / 1024. ** 2
        print(
            f"Creating 3 normal arrays of total size {size:.2f} MB on the GPU...")
        # Transfer the normal to the GPU.
        normal0_gpu = cp.asarray(normal[..., 0][box])
        normal1_gpu = cp.asarray(normal[..., 1][box])
        normal2_gpu = cp.asarray(normal[..., 2][box])

        # Create the two scalar fields.
        size = 2 * nc * mc * pc * 4 / 1024. ** 2
        print(f"Creating two arrays of total size {size:.2f} MB on the GPU...")
        Ua = cp.zeros((nc, mc, pc), dtype=np.float32)

        if U is not None:
            print(f"Starting from the existing laplacian array {U.shape}.")
            Ua[...] = cp.asarray(U[box])
        else:
            # Initial values: the same as the mask.
            Ua[...] = mask_gpu

        Ub = Ua.copy()

        # CUDA grid and block.
        b = 8
        self.block = (b, b, b)
        self.grid = (int(np.ceil(nc / float(b))),
                     int(np.ceil(mc / float(b))), int(np.ceil(pc / float(b))))

        # Main loop
        self.args = (cp.int32(nc), cp.int32(mc), cp.int32(pc))

        self.mask = mask_gpu
        self.normal0 = normal0_gpu
        self.normal1 = normal1_gpu
        self.normal2 = normal2_gpu
        self.Ua = Ua
        self.Ub = Ub
        self.box = box

    def iter(self):
        # ping-pong between the 2 arrays to avoid edge-effects while computing
        # the Laplacian in parallel on the GPU.
        # NOTE: each Python iteration here is actually made of 2 algorithm iterations
        laplace[self.grid, self.block](self.Ua, self.Ub, self.mask, *self.args)
        vonneumann[self.grid, self.block](
            self.Ub, self.mask, self.normal0, self.normal1, self.normal2, *self.args)

        laplace[self.grid, self.block](self.Ub, self.Ua, self.mask, *self.args)
        vonneumann[self.grid, self.block](
            self.Ua, self.mask, self.normal0, self.normal1, self.normal2, *self.args)

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
        del self.mask
        del self.Ua
        del self.Ub
        del self.normal0_gpu
        del self.normal1_gpu
        del self.normal2_gpu
        clear_gpu_memory()


def compute_laplacian():

    mask = get_mask(REGION)
    assert mask.ndim == 3
    assert mask.shape == (N, M, P)

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
    compute_laplacian()

    # U = load_npy(filepath(REGION, 'laplacian'))

    # i0 = 500
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # # Colormap scaling.
    # # Umin, Umax = U.min(), U.max()
    # q = .001
    # Umin = np.quantile(U.ravel(), q)
    # Umax = np.quantile(U.ravel(), 1-q)
    # norm = Normalize(vmin=Umin, vmax=Umax)

    # ims = ax.imshow(U[i0, :, :], interpolation='none',
    #                 origin='lower', norm=norm)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(ims, cax=cax, orientation='vertical')

    # axs = plt.axes([0.25, 0.1, 0.65, 0.03])
    # slider = Slider(
    #     ax=axs,
    #     label="i",
    #     valmin=0,
    #     valstep=1,
    #     valmax=N-1,
    #     valinit=i0,
    #     orientation="horizontal"
    # )

    # @slider.on_changed
    # def update(i):
    #     i = int(i)
    #     ims.set_data(U[i, :, :])
    #     fig.canvas.draw_idle()

    # plt.show()
