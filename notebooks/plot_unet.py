import torch

from metrics.evaluation import compute_all_metrics
from plotting import plot_recon_3d

pcs = torch.load("results/unet/pc.pt").cpu()
refs = torch.load("results/encoder_airplane/references.pt")

plot_recon_3d(pcs, refs, num_pc=10, offset=20)
