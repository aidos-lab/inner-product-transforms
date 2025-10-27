import torch

from metrics.evaluation import compute_all_metrics
from plotting import plot_recon_3d

pcs = torch.load("results/scratch/recon_sv.pt").cpu()
import pyvista as pv

pv.plot(pcs.squeeze().numpy())


# refs = torch.load("results/encoder_airplane/references.pt")

# plot_recon_3d(pcs, pcs, num_pc=10, offset=20)
