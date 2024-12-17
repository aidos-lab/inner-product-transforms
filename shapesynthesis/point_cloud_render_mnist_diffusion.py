from sympy import degree
import torch
from layers.directions import generate_uniform_directions
import pyvista as pv
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud
from datasets.mnist import DataModule, DataModuleConfig
from layers.ect import EctConfig
from plotting import plot_recon_2d
import matplotlib.pyplot as plt


NUM_EPOCHS = 2000
RESOLUTION = 64
SCALE = 48
SEED = 2024
RADIUS = 1.1
DTYPE = torch.float32
NUM_PTS = 64
DIM = 2

v = (
    generate_uniform_directions(num_thetas=RESOLUTION, d=DIM, seed=SEED)
    .type(DTYPE)
    .cuda()
)

ects = torch.load("./shapesynthesis/diff.pt").cuda() * NUM_PTS
# ects = torch.load("./ectbatch.pt").cuda() * NUM_PTS
print(ects.min())
print(ects.max())
print(len(ects))

x_init = (torch.rand(size=(len(ects), NUM_PTS, DIM), dtype=DTYPE) - 0.5).cuda()

x_rendered = render_point_cloud(
    x_init,
    ects,
    v,
    NUM_EPOCHS,
    scale=SCALE,
    radius=RADIUS,
    resolution=RESOLUTION,
)

print(x_rendered.shape)


DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 5, "c": ".5"}
LIGHTRED = [255, 100, 100]
import numpy as np


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(8 * 2, 2 * 2))

for recon_pc, ect, axis in zip(x_rendered[8:], ects[8:], axes.T):
    recon_pc = rotate(recon_pc.squeeze().numpy(), degrees=-90)
    ax = axis[0]
    ax.scatter(recon_pc[:, 0], recon_pc[:, 1], **PC_PLOT_CONFIG)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect(1)
    ax.axis("off")

    ax = axis[1]
    ax.imshow(ect.squeeze().cpu().numpy())
    ax.set_aspect(1)
    ax.axis("off")

plt.show()
