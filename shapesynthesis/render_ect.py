import torch
from layers.directions import generate_uniform_directions
import pyvista as pv
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud
from datasets.mnist import DataModule, DataModuleConfig
from layers.ect import EctConfig
from plotting import plot_recon_2d, plot_recon_3d
import matplotlib.pyplot as plt
import numpy as np

from loaders import load_config


NUM_EPOCHS = 2000
# RESOLUTION = 64
# SCALE = 128
# SEED = 2024
# RADIUS = 1.1
# DTYPE = torch.float32


NUM_PTS = 1024


config, _ = load_config("./shapesynthesis/configs/vae_chair.yaml")
ectconfig = config.data.ectconfig
print(ectconfig)
v = generate_uniform_directions(
    num_thetas=ectconfig.num_thetas, d=ectconfig.ambient_dimension, seed=ectconfig.seed
).cuda()

# Load the ECT's
ects = torch.load("./results/vae_chair/ect.pt").cuda()[:2] * NUM_PTS
# ects = torch.load("./ectbatch.pt").cuda() * NUM_PTS
print("ECT min", ects.min())
print("ECT max", ects.max())
print(len(ects))

x_init = (
    torch.rand(size=(len(ects), NUM_PTS, ectconfig.ambient_dimension)) - 0.5
).cuda() * 5

x_rendered = render_point_cloud(
    x_init,
    ects,
    v,
    NUM_EPOCHS,
    scale=ectconfig.scale,
    radius=ectconfig.r,
    resolution=ectconfig.resolution,
)


if ectconfig.ambient_dimension == 2:
    fig = plot_recon_2d(x_rendered, x_rendered)
    plt.show()
else:
    plot_recon_3d(
        x_rendered.detach().cpu().numpy(), x_rendered.detach().cpu().numpy(), num_pc=2
    )

# DEVICE = "cuda:0"
# ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
# PC_PLOT_CONFIG = {"s": 5, "c": ".5"}
# LIGHTRED = [255, 100, 100]


# def rotate(p, origin=(0, 0), degrees=0):
#     angle = np.deg2rad(degrees)
#     R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#     o = np.atleast_2d(origin)
#     p = np.atleast_2d(p)
#     return np.squeeze((R @ (p.T - o.T) + o.T).T)


# fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(8 * 2, 2 * 2))

# for recon_pc, ect, axis in zip(x_rendered[8:], ects[8:], axes.T):
#     recon_pc = rotate(recon_pc.squeeze().numpy(), degrees=-90)
#     ax = axis[0]
#     ax.scatter(recon_pc[:, 0], recon_pc[:, 1], **PC_PLOT_CONFIG)
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_aspect(1)
#     ax.axis("off")

#     ax = axis[1]
#     ax.imshow(ect.squeeze().cpu().numpy())
#     ax.set_aspect(1)
#     ax.axis("off")

# plt.show()
