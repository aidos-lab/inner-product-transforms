"""
Evaluation script for the generated ECT's from the
VAE. We calculate the FID Score between the generated
images and the training set and the test set.
"""

import torch
from torchvision.transforms import Resize
from torchmetrics.image.fid import FrechetInceptionDistance

# Load the images into memory from the results folder.
# The "latent" postfix are the results used in the paper.
sample_ect = torch.load("./results/vae_airplane_latent/sample_ect.pt").cpu()
reference_ect = torch.load("./results/vae_airplane_latent/ect.pt").cpu()

# Load the test images, live in a different folder.

# Print data types etc for sanity checking.
print(reference_ect.min().item(), reference_ect.max().item())
print(sample_ect.min().item(), sample_ect.max().item())

t = Resize(299)
resized_ect = (
    t(255 * sample_ect).to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1),
)
resized_ref_ect = (
    t(255 * reference_ect).to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1),
)


fid = FrechetInceptionDistance(feature=64)
fid.update(resized_ref_ect, real=True)
fid.update(resized_ect, real=False)
f = fid.compute()
print(f)
