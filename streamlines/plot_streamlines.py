import time
import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap


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
    color = colormap(np.arange(n, dtype=np.double), vmin=0,
                     vmax=1, cmap='viridis', alpha=1)
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


paths_allen = np.load("resampled_allen.npy", mmap_mode='r')
paths_ibl = np.load("resampled_ibl.npy", mmap_mode='r')

n = 100000
paths_allen = np.array(paths_allen[::int(paths_allen.shape[0] // n), ...][:n])
paths_ibl = np.array(
    paths_ibl[::int(paths_ibl.shape[0] // n), ...][:n][:, ::-1, :])

assert paths_allen.shape[0] == n
assert paths_ibl.shape[0] == n

l = 100
length = l * np.ones(n)  # length of each path


color = np.tile(np.linspace(0, 1, l), n)
color = colormap(color, vmin=0, vmax=1, cmap='viridis', alpha=.75)

c = canvas(show_fps=True)
s = c.scene(cols=2)


# Left panel.
# ------------
p0 = s.panel(col=0, controller='arcball')

# v0 = p0.visual('line_strip', depth_test=True)
# v0.data('pos', paths_allen.reshape((-1, 3)))
# v0.data('length', length)
# v0.data('color', color)

# Points.
v0p = p0.visual('point', depth_test=True)
v0p.data('pos', paths_allen[:, 0, :])
v0p.data('color', np.array([[255, 0, 0, 128]]))


# Right panel.
# ------------
p1 = s.panel(col=1, controller='arcball')

# v1 = p1.visual('line_strip', depth_test=True)
# v1.data('pos', paths_ibl.reshape((-1, 3)))
# v1.data('length', length)
# v1.data('color', color)

# Points.
v1p = p1.visual('point', depth_test=True)
v1p.data('pos', paths_ibl[:, 0, :])
v1p.data('color', np.array([[255, 0, 0, 128]]))


# Panel linking.
p0.link_to(p1)

run()
