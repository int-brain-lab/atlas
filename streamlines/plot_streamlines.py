import time
import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap


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
