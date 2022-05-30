#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *

from scipy.interpolate import interpn
from scipy.interpolate import interp1d


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

REGION = 'isocortex'
REGION_ID = 315
N, M, P = 1320, 800, 1140
PATH_LEN = 100
MAX_POINTS = 100000
MAX_ITER = 500
STEP = .5


# ------------------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------------------

def last_nonzero(arr, axis, invalid_val=-1):
    # https://stackoverflow.com/a/47269413/1595060
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def subset(paths, max_paths=None):
    if not max_paths:
        return paths
    n = paths.shape[0]
    k = max(1, int(math.floor(float(n) / float(max_paths))))
    return np.array(paths[::k, ...])


# ------------------------------------------------------------------------------------------------
# Gradient
# ------------------------------------------------------------------------------------------------

def compute_grad(mask, U):
    n, m, p = mask.shape

    # Find the surface.
    i, j, k = np.nonzero(np.isin(mask, (V_S1, V_S2, V_Si)))
    surf = np.zeros((n, m, p), dtype=bool)
    surf[i, j, k] = True
    iv, jv, kv = np.nonzero(mask == V_VOLUME)

    # Clip the laplacian.
    q = .9999
    Uclip = np.clip(U, U.min(), np.quantile(U, q))

    # Compute the gradient inside the volume.
    grad = np.zeros((n, m, p, 3), dtype=np.float32)
    grad[iv, jv, kv, 0] = .5 * (Uclip[iv+1, jv, kv] - Uclip[iv-1, jv, kv])
    grad[iv, jv, kv, 1] = .5 * (Uclip[iv, jv+1, kv] - Uclip[iv, jv-1, kv])
    grad[iv, jv, kv, 2] = .5 * (Uclip[iv, jv, kv+1] - Uclip[iv, jv, kv-1])

    # Compute the gradient on the surface.
    idx = mask[i+1, j, k] == V_VOLUME
    grad[i[idx], j[idx], k[idx], 0] = Uclip[
        i[idx]+1, j[idx], k[idx]] - Uclip[i[idx], j[idx], k[idx]]

    idx = mask[i-1, j, k] == V_VOLUME
    grad[i[idx], j[idx], k[idx], 0] = Uclip[
        i[idx], j[idx], k[idx]] - Uclip[i[idx]-1, j[idx], k[idx]]

    idx = mask[i, j+1, k] == V_VOLUME
    grad[i[idx], j[idx], k[idx], 1] = Uclip[
        i[idx], j[idx]+1, k[idx]] - Uclip[i[idx], j[idx], k[idx]]

    idx = mask[i, j-1, k] == V_VOLUME
    grad[i[idx], j[idx], k[idx], 1] = Uclip[
        i[idx], j[idx], k[idx]] - Uclip[i[idx], j[idx]-1, k[idx]]

    idx = mask[i, j, k+1] == V_VOLUME
    grad[i[idx], j[idx], k[idx], 2] = Uclip[
        i[idx], j[idx], k[idx]+1] - Uclip[i[idx], j[idx], k[idx]]

    idx = mask[i, j, k-1] == V_VOLUME
    grad[i[idx], j[idx], k[idx], 2] = Uclip[
        i[idx], j[idx], k[idx]] - Uclip[i[idx], j[idx], k[idx]-1]

    return grad


def normalize_gradient(grad, threshold=0):
    # Normalize the gradient.
    gradn = np.linalg.norm(grad, axis=3)

    idx = gradn > threshold
    grad[idx] /= gradn[idx, np.newaxis]

    # Kill gradient vectors that are too small.
    if threshold > 0:
        grad[~idx] = 0

    return grad


def get_gradient(region):
    path = filepath(region, 'gradient_allen')
    gradient = load_npy(path)
    if gradient is not None:
        return gradient

    # Load the laplacian to compute the gradient.
    U = load_npy(filepath(region, 'laplacian_allen'))
    if U is None:
        # TODO: compute the laplacian with code in streamlines.py
        raise NotImplementedError()
    assert U.ndim == 3

    # Load the mask.
    mask = load_npy(filepath(region, 'mask'))

    # Compute the gradient.
    gradient = compute_grad(mask, U)
    assert gradient.ndim == 4

    # Normalize the gradient.
    gradient = normalize_gradient(gradient)

    # Save the gradient.
    save_npy(path, gradient)

    del gradient
    return load_npy(path)


# ------------------------------------------------------------------------------------------------
# Initial points (seeds)
# ------------------------------------------------------------------------------------------------

def init_allen(region):
    return load_npy(filepath(region, 'streamlines_allen'))[:, 0, :]


def init_ibl(region):
    mask = get_mask(region)
    assert mask.ndim == 3
    i, j, k = np.nonzero(np.isin(mask, [V_S2]))
    pos = np.c_[i, j, k]
    return pos


# ------------------------------------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------------------------------------

def integrate_step(pos, step, gradient, xyz):
    assert pos.ndim == 2
    assert pos.shape[1] == 3
    assert gradient.shape == (N, M, P, 3)
    for i in range(3):
        pos[:, i] = np.clip(pos[:, i], xyz[i][0], xyz[i][-1])
    g = interpn(xyz, gradient, pos)
    return pos + step * g


def integrate_field(pos, step, gradient, mask, max_iter=MAX_ITER, res_um=0):
    # , stay_in_volume=False):
    assert pos.ndim == 2
    n_paths = pos.shape[0]
    assert pos.shape == (n_paths, 3)

    n, m, p = mask.shape
    res_um = 1
    x = np.arange(0, res_um * n, res_um)
    y = np.arange(0, res_um * m, res_um)
    z = np.arange(0, res_um * p, res_um)
    xyz = (x, y, z)

    out = np.zeros((n_paths, max_iter, 3), dtype=np.float32)
    out[:, 0, :] = pos
    pos_grid = np.zeros((n_paths, 3), dtype=np.int32)

    # Which positions are still in the volume and need to be integrated?
    kept = slice(None, None, None)

    for iter in tqdm(range(1, max_iter), desc="Integrating..."):
        prev = out[kept, iter - 1, :]
        out[kept, iter, :] = integrate_step(prev, step, gradient, xyz)
        # if not stay_in_volume:
        #     continue

        # # Stop integrating the paths the go outside of the volume.
        # # get the masks on the current positions
        # pos_grid[:] = out[:, iter, :]
        # i, j, k = pos_grid.T
        # kept = mask[i, j, k] != 0
        # assert kept.shape == (n_paths,)
        # n_kept = kept.sum()
        # if n_kept == 0:
        #     break

    return out


def path_lengths(paths):
    print("Computing the path lengths...")
    streamlines = paths
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


def compute_streamlines(region, init_points=None):

    # Load the region mask.
    mask = get_mask(region)
    assert mask.ndim == 3

    # Download or load the mesh (initial positions of the streamlines).
    if init_points is None:
        init_points = init_ibl(region)
    assert init_points.ndim == 2
    assert init_points.shape[1] == 3
    n = len(init_points)
    init_points = subset(init_points, MAX_POINTS)
    print(f"Starting computing {len(init_points)} (out of {n}) streamlines...")

    # Compute or load the gradient.
    gradient = get_gradient(region)
    assert gradient.ndim == 4
    assert gradient.shape[3] == 3

    # Integrate the gradient field from those positions.
    paths = integrate_field(
        init_points, STEP, gradient, mask, max_iter=MAX_ITER)

    # Resample the paths.
    streamlines = resample_paths(paths, num=PATH_LEN)

    # Save the streamlines.
    save_npy(filepath(region, 'streamlines_ibl'), streamlines)


if __name__ == '__main__':
    get_gradient(REGION)
    # compute_streamlines(REGION)
