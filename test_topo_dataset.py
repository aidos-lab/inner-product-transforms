import numpy as np
import torch
import pyvista as pv

from datasets.topological import (
    TopolocigalDataModule,
    TopologicalDataModuleConfig,
)
from torch_topological.data import sample_from_sphere
from torch_topological.data import sample_from_torus, sample_from_unit_cube


dm = TopolocigalDataModule(TopologicalDataModuleConfig())

for fullbatch in dm.train_dataloader():
    break

# mobius = torch.stack([sample_from_torus(10) for i in range(30)])

# data = torch.vstack((mobius, mobius))
# print(data.shape)


data = torch.load("./data/topological/raw/raw_train.pt")
print(data.shape)

# pl = pv.Plotter(shape=(8, 8), window_size=[800, 800])

# for batch_idx in range(8):
#     batch = fullbatch[8 * batch_idx : :].cpu().etach().numpy()d

#     for idx in range(8):
#         points = batch[idx].reshape(-1, 3)
#         pl.subplot(batch_idx, idx)
#         actor = pl.add_points(
#             points,
#             style="points",
#             emissive=False,
#             show_scalar_bar=False,
#             render_points_as_spheres=True,
#             scalars=points[:, 2],
#             point_size=10,
#             ambient=0.2,
#             diffuse=0.8,
#             specular=0.8,
#             specular_power=40,
#             smooth_shading=True,
#         )


# pl.background_color = "k"
# pl.link_views()
# pl.camera_position = "yz"
# pos = pl.camera.position
# pl.camera.position = (pos[0], pos[1], pos[2] + 3)
# pl.camera.azimuth = -45
# pl.camera.elevation = 10

# # create a top down light
# light = pv.Light(
#     position=(0, 0, 3), positional=True, cone_angle=50, exponent=20, intensity=0.2
# )
# pl.add_light(light)
# pl.camera.zoom(0.8)
# pl.show()
