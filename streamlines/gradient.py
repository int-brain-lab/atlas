#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *

from scipy.interpolate import interpn
from scipy.interpolate import interp1d


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
    path = filepath(region, 'gradient')
    gradient = load_npy(path)
    if gradient is not None:
        return gradient

    # Load the laplacian to compute the gradient.
    U = load_npy(filepath(region, 'laplacian'))
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


if __name__ == '__main__':
    get_gradient(REGION)
