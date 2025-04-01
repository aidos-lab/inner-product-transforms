"""
Script to generate a render of an object and records all steps. 
The intermediary points are stored in the results folder.
"""

import torch

from loaders import load_config, load_datamodule
from layers.directions import generate_uniform_directions
from renderers.pointcloud import render_point_cloud_viz


# ############################################################################ #
#                                   Constants                                  #
# ############################################################################ #

# --------------------------------- Constants -------------------------------- #


CATE = "airplane"
IDX = 17  # The ect idx to load
# Number of epochs to run the point cloud optimization for. Default used is
# 2000.
NUM_EPOCHS = 2000

# ---------------------------------- Loading --------------------------------- #

# Load the configuration of the car encoder model
print("Loading config")
config, _ = load_config(f"./shapesynthesis/configs/encoder_{CATE}.yaml")
ectconfig = config.data.ectconfig

# Generate the directions.
v = generate_uniform_directions(
    num_thetas=ectconfig.num_thetas,
    d=ectconfig.ambient_dimension,
    seed=ectconfig.seed,
).cuda()

# Load the ECT's
print("Loading ect")

dm = load_datamodule(config.data)
ect = dm.val_ds[IDX].ect * config.modelconfig.num_pts

# Basic bookkeeping for reference
print("Min", ect[0].min())
print("Max", ect[0].max())

print("Done")

# # ############################################################################ #
# #                         Run the rendering script                             #
# # ############################################################################ #

# Initialize a random point cloud
x_init = (
    torch.rand(size=(1, config.modelconfig.num_pts, ectconfig.ambient_dimension)) - 0.5
).cuda()


# Run the rendering script.
x_rendered, result = render_point_cloud_viz(
    x_init,
    ect.view(1, 256, 256).cuda(),
    v,
    NUM_EPOCHS,
    scale=ectconfig.scale,
    radius=ectconfig.r,
    resolution=ectconfig.resolution,
)

# Save the results.
torch.save(result, f"./results/rendered_ect/{CATE}/full_orbit.pt")
