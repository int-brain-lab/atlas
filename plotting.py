"""Plotting streamlines in 3D with Datoviz."""

import numpy as np

from common import *
from streamlines import *

from datoviz import canvas, run, colormap


def plot_panel(panel, paths):
    assert paths.ndim == 3
    n, l, _ = paths.shape
    assert _ == 3
    length = l * np.ones(n)  # length of each path

    color = np.tile(np.linspace(0, 1, l), n)
    color = colormap(color, vmin=0, vmax=1, cmap='viridis', alpha=1)

    v = panel.visual('line_strip', depth_test=True)
    v.data('pos', paths.reshape((-1, 3)))
    v.data('length', length)
    v.data('color', color)


paths = load_npy(filepath(REGION, 'streamlines'))

# Subset the paths.
max_paths = 100_000
paths = subset(paths, max_paths)

paths = paths[:, ::-1, :]

c = canvas(show_fps=True)
s = c.scene(cols=1)
p = s.panel(controller='arcball')
plot_panel(p, paths)

run()
