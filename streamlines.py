#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *
from gradient import *

from scipy.interpolate import interpn
from scipy.interpolate import interp1d


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

PATH_LEN = 100
MAX_POINTS = None
MAX_ITER = 500
STEP = 1.0


# ------------------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------------------

def last_nonzero(arr, axis, invalid_val=-1):
    """Return the index of the last vector element with a zero."""

    # https://stackoverflow.com/a/47269413/1595060
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def subset(paths, max_paths=None):
    """Get a subset of all paths."""

    if not max_paths:
        return paths
    n = paths.shape[0]
    k = max(1, int(math.floor(float(n) / float(max_paths))))
    return np.array(paths[::k, ...])


# ------------------------------------------------------------------------------------------------
# Initial points (seeds)
# ------------------------------------------------------------------------------------------------

def init_allen(region):
    """Use the initial points of the Allen streamlines."""
    return load_npy(filepath(region, 'streamlines_allen'))[:, 0, :]


def init_ibl(region):
    """Use the voxels in the top surface as initial points for the streamlines."""
    mask = get_mask(region)
    assert mask.ndim == 3
    i, j, k = np.nonzero(np.isin(mask, [V_ST]))
    pos = np.c_[i, j, k]

    # HACK: fix bug with large empty areas when plotting streamlines. For some reason,
    # taking a subset of the positions by slicing pos[::k, :] results in large empty areas
    # in one of the hemispheres for some values of k (even values). We fix this systematic bias by
    # shuffling the initial positions.
    np.random.seed(0)
    perm = np.random.permutation(pos.shape[0])

    return pos[perm, :]


# ------------------------------------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------------------------------------

def integrate_step(pos, step, gradient, xyz):
    """Run one step of the integration process."""

    assert pos.ndim == 2
    assert pos.shape[1] == 3
    assert gradient.shape == (N, M, P, 3)
    for i in range(3):
        pos[:, i] = np.clip(pos[:, i], xyz[i][0], xyz[i][-1])
    g = interpn(xyz, gradient, pos)
    # NOTE: - if bottom to top, + if top to bottom
    return pos + step * g


def integrate_field(pos, step, gradient, target, max_iter=MAX_ITER):
    """Generate streamlines."""

    assert pos.ndim == 2
    n_paths = pos.shape[0]
    assert pos.shape == (n_paths, 3)

    n, m, p = target.shape
    res_um = 1
    x = np.arange(0, res_um * n, res_um)
    y = np.arange(0, res_um * m, res_um)
    z = np.arange(0, res_um * p, res_um)
    xyz = (x, y, z)

    out = np.zeros((n_paths, max_iter, 3), dtype=np.float32)
    out[:, 0, :] = pos
    pos_grid = np.zeros((n_paths, 3), dtype=np.int32)

    # Which positions are still in the volume and need to be integrated?
    kept = np.ones(n_paths, dtype=bool)

    with tqdm(range(1, max_iter), desc="Integrating...") as t:
        for iter in t:
            # Previous position of the kept streamlines.
            prev_pos = out[kept, iter - 1, :]

            # Compute the new positions.
            new_pos = integrate_step(prev_pos, step, gradient, xyz)

            # Save the new positions in the output array.
            out[kept, iter, :] = new_pos

            # For the dropped streamlines, we copy the same values.
            out[~kept, iter, :] = out[~kept, iter - 1, :]

            # Stop integrating the paths the go outside of the volume.
            # Convert the new positions to indices.
            i, j, k = np.round(new_pos).astype(np.int32).T
            i[:] = np.clip(i, 0, n - 1)
            j[:] = np.clip(j, 0, m - 1)
            k[:] = np.clip(k, 0, p - 1)
            # i, j, k are indices of the streamlines within the volume.

            # The paths that are kept are the paths that have not reached the target yet.
            kept[kept] = target[i, j, k] == 0

            assert kept.shape == (n_paths,)
            n_kept = kept.sum()
            t.set_postfix(n_kept=n_kept)
            if n_kept == 0:
                break

    return out


def path_lengths(paths):
    """Compute the lengths of the streamlines."""

    print("Computing the path lengths...")
    streamlines = paths
    n_paths, path_len, _ = streamlines.shape
    d = (np.diff(paths, axis=1) ** 2).sum(axis=2) > 1e-5
    # d = np.abs(np.diff(paths, axis=1)).max(axis=2) > 1e-3
    ln = last_nonzero(d, 1)
    assert ln.shape == (n_paths,)
    return ln


def resample_paths(paths, num=PATH_LEN):
    """Resample the streamlines."""

    n_paths, path_len, _ = paths.shape
    xp = np.linspace(0, 1, num)

    lengths = path_lengths(paths)
    # HACK: length == -1 means that the path has not reached the target yet so we resample all
    # of it
    lengths[lengths < 0] = path_len

    out = np.zeros((n_paths, num, 3), dtype=np.float32)
    for i in tqdm(range(n_paths), desc="Resampling..."):
        n = lengths[i]
        if n >= 2:
            lin = interp1d(np.linspace(0, 1, n), paths[i, :n, :], axis=0)
            out[i, :, :] = lin(xp)
        else:
            out[i, :, :] = paths[i, :num, :]
    return out


def compute_streamlines(region, init_points=None):
    """Compute (or load from the cache) the streamlines."""

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

    # Stop the integration when points reach the bottom surface.
    target = np.isin(mask, (V_SB,))

    # Integrate the gradient field from those positions.
    paths = integrate_field(
        init_points, STEP, gradient, target, max_iter=MAX_ITER)

    # Resample the paths.
    streamlines = resample_paths(paths, num=PATH_LEN)

    # Save the streamlines.
    save_npy(filepath(region, 'streamlines'), streamlines)


# ------------------------------------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    compute_streamlines(REGION)
