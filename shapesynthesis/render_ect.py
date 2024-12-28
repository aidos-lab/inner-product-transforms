import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from datasets.mnist import DataModule, DataModuleConfig
from layers.directions import generate_uniform_directions
from layers.ect import EctConfig, compute_ect_point_cloud
from loaders import load_config
from plotting import plot_recon_2d, plot_recon_3d
from renderers.pointcloud import render_point_cloud

NUM_EPOCHS = 2000

NUM_PTS = 64

config, _ = load_config("./shapesynthesis/configs/vae_mnist.yaml")
ectconfig = config.data.ectconfig
# ectconfig.scale = 32
print(ectconfig)


v = generate_uniform_directions(
    num_thetas=ectconfig.num_thetas,
    d=ectconfig.ambient_dimension,
    seed=ectconfig.seed,
).cuda()

# Load the ECT's
ects = (
    torch.load("./results/vae_mnist/reconstructed_ect.pt").cuda()[:9] * NUM_PTS
)
# ects = torch.load("./ectbatch.pt").cuda() * NUM_PTS
print("ECT min", ects.min())
print("ECT max", ects.max())
print(len(ects))
plt.imshow(ects[0].squeeze().cpu().numpy())
plt.show()

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
        x_rendered.detach().cpu().numpy(),
        x_rendered.detach().cpu().numpy(),
        num_pc=2,
    )
