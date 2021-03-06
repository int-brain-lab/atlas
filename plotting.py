"""Plotting streamlines in 3D with Datoviz."""

import numpy as np

from common import *
from streamlines import *

from datoviz import canvas, run, colormap


SHOW_ALLEN = True
MAX_PATHS = 100_000


def ibl_streamlines():
    paths = load_npy(filepath(REGION, 'streamlines'))
    paths = subset(paths, MAX_PATHS)

    # NOTE: the following line should be decommented when streamlines go from bottom to top
    # paths = paths[:, ::-1, :]

    return paths


def allen_streamlines():
    paths = load_npy(filepath(REGION, 'streamlines_allen'))
    paths = subset(paths, MAX_PATHS)
    return paths


def plot_panel(panel, paths):
    assert paths.ndim == 3
    n, l, _ = paths.shape
    assert _ == 3
    length = l * np.ones(n)  # length of each path

    color = np.tile(np.linspace(0, 1, l), n)
    color = colormap(color, vmin=0, vmax=1, cmap='viridis', alpha=1)

    # Plot lines
    v = panel.visual('line_strip', depth_test=True)
    v.data('pos', paths.reshape((-1, 3)))
    v.data('length', length)
    v.data('color', color)

    # # Plot points
    # v = panel.visual('point', depth_test=True)
    # v.data('pos', paths.reshape((-1, 3)))
    # v.data('ms', np.array([1.0]))
    # v.data('color', color)


c = canvas(show_fps=True)
s = c.scene(cols=2 if SHOW_ALLEN else 1)

if SHOW_ALLEN:
    paths_allen = allen_streamlines()
    p_allen = s.panel(col=0, controller='arcball')
    plot_panel(p_allen, paths_allen)

paths_ibl = ibl_streamlines()
p_ibl = s.panel(col=1 if SHOW_ALLEN else 0, controller='arcball')
plot_panel(p_ibl, paths_ibl)
if SHOW_ALLEN:
    p_allen.link_to(p_ibl)

run()
