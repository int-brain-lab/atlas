#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------------------------
# Normal smoothing (Gaussian convolution)
# ------------------------------------------------------------------------------------------------


"""
Masked convolution:

$$
\widetilde x[i_0,j_0,k_0,d] = \frac{
    \sum_{i,j,k} x[i_0-i, j_0-j, k_0-k,d] \cdot m[i_0-i, j_0-j, k_0-k] \cdot g[i,j,k]}
    {\sum_{i,j,k} m[i_0-i, j_0-j, k_0-k] \cdot g[i,j,k]}
$$
"""


def gaussian_kernel(size, sigma):
    """Return N-Dimensional Gaussian Kernel
    @param integer  size  size of kernel / will be round to a nearest odd number
    @param float    sigma standard deviation of gaussian
    https://gist.github.com/tohki/e8803620c2abaa2083f6
    """
    assert size % 2 == 0
    s = int(size // 2)
    x, y, z = np.mgrid[-s:s, -s:s, -s:s]
    k = np.exp(-(np.power(x, 2) + np.power(y, 2) + np.power(z, 2)) / (2 * (sigma ** 2)))
    return (k / k.sum()).astype(np.float32)


@numba.njit()
def _convol(arrp, maskp, surf_idx=None, gauss=None):
    # NOTE: arrp and maskp must be padded already
    assert gauss is not None

    width = w = gauss.shape[0]
    hw = w // 2
    ni, nj, nk, nd = arrp.shape

    # HACK: NO PADDING because Python crashes with np.pad() on large arrays
    w = 0

    ni -= 2*w
    nj -= 2*w
    nk -= 2*w

    out = np.zeros((ni, nj, nk, nd), dtype=np.float32)
    x = 0
    n = len(surf_idx)
    print(f"Starting the convolution")
    for iter in range(n):
        if iter % 100000 == 0:
            print(100 * iter / float(n))
        i0, j0, k0 = surf_idx[iter]

        # HACK: NO PADDING
        assert hw <= i0 and i0+hw < ni and hw <= j0 and j0+hw < nj and hw <= k0 and k0+hw < nk

        sl = (
            slice(w+i0-hw, w+i0+hw, None),
            slice(w+j0-hw, w+j0+hw, None),
            slice(w+k0-hw, w+k0+hw, None)
        )
        assert maskp[i0+w, j0+w, k0+w]
        for d in range(nd):
            mg = maskp[sl] * gauss
            su = np.sum(mg)
            if su != 0:
                x = np.sum(arrp[sl + (d,)] * mg)
                x /= su
                out[i0, j0, k0, d] = x
    return out


def convol(arr, surf_mask, surf_idx=None, width=6, sigma=1.0):

    assert arr.ndim == 4
    assert arr.dtype == np.int8
    assert surf_mask.ndim == 3
    assert surf_mask.dtype == bool
    assert surf_mask.shape == arr.shape[:3]

    assert width % 2 == 0

    if surf_idx is None:
        i, j, k = np.nonzero(surf_mask > 0)
        surf_idx = np.vstack((i, j, k)).T

    assert surf_idx.ndim == 2
    assert surf_idx.shape[1] == 3
    assert surf_idx.dtype == np.int64

    print(f"Computing the Gaussian kernel")
    gauss = gaussian_kernel(width, sigma)
    assert gauss.shape == (width, width, width)

    print(f"Padding the arrays")
    # These functions crash on the big volumes
    #arrp = np.pad(arr, width, mode='edge')
    #maskp = np.pad(surf_mask, width, mode='edge')
    arrp = arr
    maskp = surf_mask

    return _convol(arrp, maskp, surf_idx=surf_idx, gauss=gauss)


def normalize_normal(normal):
    assert normal.ndim == 4
    assert normal.shape[3] == 3

    norm = np.linalg.norm(normal, axis=3)
    idx = norm > 0
    normal[idx] /= norm[idx][..., np.newaxis]

    return normal


def get_normal(region):
    path = filepath(region, 'normal')
    if path.exists():
        return load_npy(path)

    print(f"Computing surface normal...")
    mask = get_mask(region)
    normal = compute_normal(mask)
    assert normal.ndim == 4

    surf_vals = (V_S1, V_S2, V_Si)
    surface_mask = get_surface_mask(region, surf_vals)
    assert surface_mask.ndim == 3

    surf_indices = get_surface_indices(region, surf_vals)
    assert surf_indices.ndim == 2
    assert surf_indices.shape[1] == 3
    width = 8
    sigma = 2.0
    normal_smooth = convol(normal, surface_mask, surf_idx=surf_indices, width=width, sigma=sigma)

    # Save the normal file.
    save_npy(path, normal_smooth)
    return load_npy(path)


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    normal = get_normal(REGION)

    i0 = 500
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ims = ax.imshow(normal[i0, :, :, 0], interpolation='none', origin='upper')

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
        ims.set_data(normal[i, :, :, 0])
        fig.canvas.draw_idle()

    plt.show()
