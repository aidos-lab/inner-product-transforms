"""
Script to test if the model is multimodal. Loads the ect of chair and uses the
car encoder to generate the point clouds.

It is pretty evident that the model returns the chairs that are 'closest' in a 
sense to cars. In this regards theere seems to me some multimodality going on. 
"""

import torch
from loaders import load_config, load_model


# Load the configuration of the car encoder model
print("Loading config")
config, _ = load_config("./shapesynthesis/configs/encoder_chair.yaml")

# Load the encoder model.
print("Loading model")
model = load_model(config.modelconfig, "./trained_models/encoder_chair.ckpt")

# Load the ect of chairs
print("Loading ect")
ect_chairs = torch.load("./results/vae_car/ect.pt")

print("Done")

# Basic bookkeeping for reference
print(ect_chairs[0].min())
print(ect_chairs[0].max())

# Decode the chairs with the model trained on cars.

with torch.no_grad():
    recon_pc = model(ect_chairs[:10]).view(-1, 2048, 3)
print(recon_pc.shape)

import pyvista as pv

pv.plot(recon_pc[1].detach().cpu().numpy(), render_points_as_spheres=True)  # type: ignore
