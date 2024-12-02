# %%
"""Encoder module"""

import torch
from layers.ect import EctLayer, EctConfig
from layers.directions import generate_directions, generate_uniform_directions
import pyvista as pv
from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# pv.set_jupyter_backend("trame")
from datasets.shapenetcore import DataModule, DataModuleConfig
from torch_geometric.data import Data, Batch

from renderers.pointcloud import render_point_cloud


RESOLUTION = 256
NUM_EPOCHS = 500
SCALE = 100

layer = EctLayer(
    EctConfig(num_thetas=RESOLUTION, bump_steps=RESOLUTION, r=5),
    v=generate_uniform_directions(RESOLUTION, 3, seed=2024).cuda(),
).cuda()


# dm = DataModule(DataModuleConfig(cates=["chair"]))

idx = 0
# data = dm.test_ds[idx]

recon_pts = torch.load("./results/encoder_chair_sparse/reconstructions.pt")
ref_pts = torch.load("./results/encoder_chair_sparse/references.pt")

data_recon = Data(x=recon_pts[idx])
data_ref = Data(x=ref_pts[idx])

recon_batch = Batch.from_data_list([data_recon]).cuda()
ref_batch = Batch.from_data_list([data_ref]).cuda()

ect_gt = layer(ref_batch, ref_batch.batch, scale=SCALE)


x = render_point_cloud(
    ect_gt,
    layer=layer,
    num_epochs=NUM_EPOCHS,
    x_gt=ref_batch[0].x,
    x_init=recon_batch.x,
    init_radius=5,
)

pl = pv.Plotter(shape=(1, 2), window_size=[800, 400])
points = x[0].detach().cpu().view(-1, 3).numpy()
pl.subplot(0, 0)
actor = pl.add_points(
    points,
    style="points",
    emissive=False,
    show_scalar_bar=False,
    render_points_as_spheres=True,
    scalars=points[:, 2],
    point_size=5,
    ambient=0.2,
    diffuse=0.8,
    specular=0.8,
    specular_power=40,
    smooth_shading=True,
)
pl.subplot(0, 1)
points = ref_batch.x.reshape(-1, 3).detach().cpu().numpy()
actor = pl.add_points(
    points,
    style="points",
    emissive=False,
    show_scalar_bar=False,
    render_points_as_spheres=True,
    scalars=points[:, 2],
    point_size=5,
    ambient=0.2,
    diffuse=0.8,
    specular=0.8,
    specular_power=40,
    smooth_shading=True,
)

pl.background_color = "w"
pl.link_views()
pl.camera_position = "xy"
pos = pl.camera.position
pl.camera.position = (pos[0] + 3, pos[1], pos[2])
pl.camera.azimuth = 145
pl.camera.elevation = 20

# create a top down light
light = pv.Light(
    position=(0, 0, 3), positional=True, cone_angle=50, exponent=20, intensity=0.2
)
pl.add_light(light)
pl.show()


import imageio

images = []
for idx in range(NUM_EPOCHS):
    if idx % 5 == 0:
        images.append(imageio.imread(f"./img/{idx}.png"))
imageio.mimsave("movie.gif", images)
