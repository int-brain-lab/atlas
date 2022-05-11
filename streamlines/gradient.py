#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from common import *

from scipy.interpolate import interpn
from scipy.interpolate import interp1d

from datoviz import canvas, run, colormap


# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

REGION = 'isocortex'
REGION_ID = 315
N, M, P = 1320, 800, 1140
OFFSET_X = 0
OFFSET_Y = 0
OFFSET_Z = 0
RES_UM = 10
PATH_LEN = 100
MAX_PATHS_PLOT = 50_000
MAX_POINTS = 20000
MAX_ITER = 2
STEP = 2000.0


# ------------------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------------------

def last_nonzero(arr, axis, invalid_val=-1):
    # https://stackoverflow.com/a/47269413/1595060
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def i2x(vol):
    """From volume indices to coordinates in microns (Allen CCF)."""
    return vol * 10.0  # + np.array([OFFSET_X, OFFSET_Y, OFFSET_Z])


def x2i(vol):
    """From coordinates in microns (Allen CCF) to volume indices."""
    return np.clip(np.floor(vol / 10.0), [0, 0, 0], [N-1, M-1, P-1]).astype(np.int32)


def subset(paths, max_paths):
    n = paths.shape[0]
    k = max(1, int(math.floor(float(n) / float(max_paths))))
    return np.array(paths[::k, ...])


# ------------------------------------------------------------------------------------------------
# Gradient
# ------------------------------------------------------------------------------------------------

def compute_grad(U):
    assert U.shape == (N, M, P)
    grad = np.stack(np.gradient(U, RES_UM, edge_order=1), axis=3)
    return grad


def normalize_gradient(grad):
    gradn = np.linalg.norm(grad, axis=3)
    idx = gradn > 0
    denom = 1. / gradn[idx]
    grad_normalized = grad.copy()
    for i in range(3):
        grad_normalized[..., i][idx] *= denom
    return grad_normalized


def clean_gradient(gradient):
    # HACK
    return gradient

    gradient = np.array(gradient, dtype=np.float32)
    gradn = np.linalg.norm(gradient, axis=3)
    assert gradn.ndim == 3
    gradn = np.repeat(gradn.reshape(gradn.shape + (1,)), 3, axis=3)
    idx = gradn > np.quantile(gradn, .99)
    gradient[idx] = 0
    return gradient


def get_gradient(region):
    # path = filepath(region, 'gradient_clean')
    # gradient = load_npy(path)
    # if gradient is not None:
    #     return gradient
    # # gradient_clean does not exist, but perhaps gradient exists
    path = filepath(region, 'gradient')
    gradient = load_npy(path)
    if gradient is not None:
        # # if it exists, clean it and save it and return it
        # gradient = clean_gradient(gradient)
        # save_npy(region, 'gradient_clean', gradient)
        return gradient
    U = load_npy(filepath(region, 'laplacian'))
    if U is None:
        # TODO: compute the laplacian with code in streamlines.py
        raise NotImplementedError()
    assert U.ndim == 3
    gradient = compute_grad(U)
    assert gradient.ndim == 4
    # save the gradient
    save_npy(filepath(region, 'gradient'), gradient)
    # # clean it
    # # gradient = clean_gradient(gradient)
    # # save it
    # save_npy(region, 'gradient_clean', gradient)
    # path = filepath(region, 'gradient_clean')
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
    i, j, k = np.nonzero(np.isin(mask, [V_S1, V_Si]))
    pos = i2x(np.c_[i, j, k])
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
    return pos - step * g


def integrate_field(pos, step, gradient, mask, max_iter=MAX_ITER, res_um=RES_UM, stay_in_volume=False):
    assert pos.ndim == 2
    n_paths = pos.shape[0]
    assert pos.shape == (n_paths, 3)

    n, m, p = mask.shape
    x = np.linspace(0, res_um * n, n) + OFFSET_X
    y = np.linspace(0, res_um * m, m) + OFFSET_Y
    z = np.linspace(0, res_um * p, p) + OFFSET_Z
    xyz = (x, y, z)

    out = np.zeros((n_paths, max_iter, 3), dtype=np.float32)
    out[:, 0, :] = pos
    pos_grid = np.zeros((n_paths, 3), dtype=np.int32)

    # Which positions are still in the volume and need to be integrated?
    kept = slice(None, None, None)

    for iter in tqdm(range(1, max_iter), desc="Integrating..."):
        prev = out[kept, iter - 1, :]
        out[kept, iter, :] = integrate_step(prev, step, gradient, xyz)
        if not stay_in_volume:
            continue

        # Stop integrating the paths the go outside of the volume.
        # get the masks on the current positions
        pos_grid[:] = x2i(out[:, iter, :])
        i, j, k = pos_grid.T
        kept = mask[i, j, k] != 0
        # print(np.bincount(mask[i, j, k]))
        assert kept.shape == (n_paths,)
        n_kept = kept.sum()
        # if iter % (int(math.ceil(max_iter / 100.0))) == 0:
        #     print(f"{n_kept}/{n_paths} remaining")
        if n_kept == 0:
            break

    return out


def path_lengths(paths):
    print("Computing the path lengths...")
    streamlines = x2i(paths)
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


def compute_streamlines(region, region_id, init_points=None):

    # Load the region mask.
    mask = get_mask(region)
    assert mask.ndim == 3

    # Download or load the mesh (initial positions of the streamlines).
    if init_points is None:
        # init_points = get_mesh(region_id, region)
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


# ------------------------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------------------------

def plot_panel(panel, paths):
    assert paths.ndim == 3
    n, l, _ = paths.shape
    assert _ == 3
    length = l * np.ones(n)  # length of each path

    color = np.tile(np.linspace(0, 1, l), n)
    color = colormap(color, vmin=0, vmax=1, cmap='viridis', alpha=1)

    # v = panel.visual('line_strip', depth_test=True)
    # v.data('pos', paths.reshape((-1, 3)))
    # v.data('length', length)
    # v.data('color', color)

    v = panel.visual('point', depth_test=True)
    v.data('pos', paths[:, 0, :].reshape((-1, 3)).astype(np.float32))
    color = colormap(np.arange(n, dtype=np.double), vmin=0, vmax=1, cmap='viridis', alpha=1)
    v.data('color', color)


def plot_streamlines(region, max_paths=MAX_PATHS_PLOT):

    paths_allen = load_npy(filepath(region, 'streamlines_allen'))
    paths_ibl = load_npy(filepath(region, 'streamlines_ibl'))

    # Subset the paths.
    paths_allen = subset(paths_allen, max_paths)
    paths_ibl = subset(paths_ibl, max_paths)

    # NOTE: convert from indices to microns
    paths_allen = (paths_allen * 10.0).astype(np.float32)

    c = canvas(show_fps=True)
    s = c.scene(cols=2)
    p0 = s.panel(col=0, controller='arcball')
    p1 = s.panel(col=1, controller='arcball')
    p0.link_to(p1)

    plot_panel(p0, paths_allen)
    plot_panel(p1, paths_ibl)

    run()


def scatter_panel(panel, points):
    assert points.ndim == 2
    assert points.shape[1] == 3
    n = points.shape[0]
    if n == 0:
        return
    colors = colormap(np.linspace(0, 1, n), vmin=0,
                      vmax=1, cmap='viridis', alpha=1)

    v = panel.visual('point', depth_test=True)
    v.data('pos', points)
    v.data('color', colors)
    v.data('ms', np.array([[5]]))


def plot_point_comparison(points0, points1):
    c = canvas(show_fps=True)
    s = c.scene(cols=2)

    p0 = s.panel(col=0, controller='arcball')
    scatter_panel(p0, points0)

    p1 = s.panel(col=1, controller='arcball')
    scatter_panel(p1, points1)

    p0.link_to(p1)

    run()


def plot_points(points):
    c = canvas(show_fps=True)
    s = c.scene()
    p = s.panel(controller='arcball')
    scatter_panel(p, points)
    run()


def plot_gradient():
    grad = get_gradient(REGION)
    k = 5
    grad = np.array(grad[::k, ::k, ::k, :], dtype=np.float32)

    gradn = np.linalg.norm(grad, axis=3)
    gradn = np.repeat(gradn.reshape(gradn.shape + (1,)), 3, axis=3)
    idx = gradn > np.quantile(gradn, .99)
    grad[idx] = 0

    x = np.arange(grad.shape[0])
    y = np.arange(grad.shape[1])
    z = np.arange(grad.shape[2])
    x, y, z = np.meshgrid(x, y, z)
    pos = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
    del x, y, z
    pos = np.repeat(pos, 2, axis=0)
    pos[1::2, :] += 100 * \
        grad.transpose((1, 0, 2, 3)).reshape(pos[::2, :].shape)
    pos = pos.reshape((-1, 2, 3))

    c = canvas(show_fps=True)
    s = c.scene()
    p = s.panel(controller='arcball')
    plot_panel(p, pos)
    run()


def plot_gradient_norm():
    grad = get_gradient(REGION)
    k = 5
    grad = np.array(grad[::k, ::k, ::k, :], dtype=np.float32)
    gradn = np.linalg.norm(grad, axis=3)

    x = np.arange(grad.shape[0])
    y = np.arange(grad.shape[1])
    z = np.arange(grad.shape[2])
    x, y, z = np.meshgrid(x, y, z)

    pos = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
    del x, y, z

    norm = gradn.transpose((1, 0, 2)).ravel()
    idx = norm > 0
    norm = norm[idx]
    pos = pos[idx]

    colors = colormap(norm.astype(np.double), cmap='viridis', alpha=1)

    c = canvas(show_fps=True)
    s = c.scene()
    p = s.panel(controller='arcball')
    v = p.visual('point', depth_test=True)

    v.data('pos', pos)
    v.data('color', colors)
    v.data('ms', np.array([[3]]))

    run()


def plot_gradient_norm_surface():
    grad = get_gradient(REGION)
    gradn = np.linalg.norm(grad, axis=3)
    assert gradn.shape == (N, M, P)

    mask = get_mask(REGION)
    assert mask.ndim == 3
    assert mask.shape == (N, M, P)

    surface = np.isin(mask, [V_S1])
    assert surface.shape == (N, M, P)
    x, y, z = np.nonzero(surface)
    x0 = x.copy()
    y0 = y.copy()
    z0 = z.copy()
    n = len(x)
    assert x.shape == y.shape == z.shape == (n,)

    pos = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
    assert pos.shape == (n, 3)

    norm = gradn[surface]
    assert norm.shape == (n,)

    colors = colormap(norm.astype(np.double), cmap='viridis', alpha=1)
    assert colors.shape == (n, 4)

    c = canvas(show_fps=True)
    s = c.scene()
    p = s.panel(controller='arcball')
    v = p.visual('point', depth_test=True)

    v.data('pos', pos)
    v.data('color', colors)
    v.data('ms', np.array([[3]]))

    u = [0, 0, 0]

    @c.connect
    def on_key_press(key, modifiers=()):
        global u, x, y, z
        if key == 'up' and 'shift' in modifiers:
            u[0] += 1
        elif key == 'down' and 'shift' in modifiers:
            u[0] -= 1
        elif key == 'up' and 'control' in modifiers:
            u[1] += 1
        elif key == 'down' and 'control' in modifiers:
            u[1] -= 1
        elif key == 'up' and 'alt' in modifiers:
            u[2] += 1
        elif key == 'down' and 'alt' in modifiers:
            u[2] -= 1
        elif key == 'r':
            u[0] = u[1] = u[2] = 0
        else:
            return
        print(u)

        x = x0 + u[0]
        y = y0 + u[1]
        z = z0 + u[2]

        norm = gradn[x, y, z]
        colors = colormap(norm.astype(np.double), cmap='viridis', alpha=1)
        v.data('color', colors)

    run()


def plot_grad_norm_hist():
    mask = get_mask(REGION)
    assert mask.ndim == 3
    i, j, k = np.nonzero(np.isin(mask, [V_S1]))

    grad = get_gradient(REGION)
    grad = np.array(grad, dtype=np.float32)
    gradn = np.linalg.norm(grad, axis=3)
    g = gradn[i, j, k]
    plt.hist(g, bins=100, log=True)
    plt.show()


if __name__ == '__main__':
    compute_streamlines(REGION, REGION_ID)
    plot_streamlines(REGION, MAX_PATHS_PLOT)
