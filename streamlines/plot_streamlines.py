import time
import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap

c = canvas(show_fps=True)
s = c.scene()

panel = s.panel(controller='arcball')
visual = panel.visual('line_strip', depth_test=True)

paths = np.load("resampled.npy")

k = 1
paths = paths[::k, ...]
n = paths.shape[0]  # number of paths

l = 100
length = l * np.ones(n)  # length of each path

color = np.tile(np.linspace(0, 1, l), n)
color = colormap(color, vmin=0, vmax=1, cmap='viridis', alpha=.9)

visual.data('pos', paths[:n, ...].reshape((-1, 3)))
visual.data('length', length)
visual.data('color', color)

run()
