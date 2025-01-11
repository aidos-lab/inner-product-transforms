"""
Interpolation.
"""

import torch
import numpy as np
from loaders import load_config, load_model

# ############################################################################ #
#                                   Constants                                  #
# ############################################################################ #

# --------------------------------- Constants -------------------------------- #

CATE = "airplane"
NUM_INTERPOLATION_STEPS = 100
START_IDX = 17  # The object to load start point
END_IDX = 19  # The object to end point

# ---------------------------------- Loading --------------------------------- #

# Load the configuration of the car encoder model
print("Loading config")
config, _ = load_config(f"./shapesynthesis/configs/encoder_{CATE}.yaml")

# Load the encoder model.
print("Loading model")
model = load_model(config.modelconfig, f"./trained_models/encoder_{CATE}.ckpt")

# Load the ect of chairs
print("Loading ect")
ect = torch.load(f"./results/vae_{CATE}/ect.pt")

# Basic bookkeeping for reference
print("Min", ect[0].min())
print("Max", ect[0].max())

print("Done")

# ############################################################################ #
#                         Run the interpolation script                         #
# ############################################################################ #

# Apply linear interpolation between the Ects
ect_start = ect[START_IDX].unsqueeze(0)
ect_end = ect[END_IDX].unsqueeze(0)

# Using torch.linspace does not work as it does not include endpoints.
interval = torch.tensor(
    np.linspace(0, 1, NUM_INTERPOLATION_STEPS, endpoint=True), dtype=torch.float32
).view(-1, 1, 1)

# Check that all shapes are correct
# [I, 256, 256] for the ECT and
# [I, 1, 1] for the interval
print("Interval Shape: ", interval.shape)
print("ECT_start shape:", ect_start.shape)
print("ECT_end shape:", ect_end.shape)

# Interval starts at 0, hence (1 - interval) goes from 1 to 0.
ect_interpolation = (1 - interval) * ect_start + interval * ect_end

with torch.no_grad():
    recon_pc = model(ect_interpolation).view(-1, 2028, 3)

torch.save(recon_pc, f"./results/interpolation/linear_{CATE}.pt")
