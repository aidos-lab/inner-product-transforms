import matplotlib.pyplot as plt
import numpy as np
import torch

from plotting import rotate

PC_PLOT_CONFIG = {"s": 4, "c": ".5"}


def plot_pc(recon_pcs, num_pc=5):

    fig, axes = plt.subplots(nrows=1, ncols=num_pc, figsize=(2 * num_pc, 2))

    for recon_pc, ax in zip(recon_pcs, axes.T):

        ax.scatter(recon_pc[:, 0], recon_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")
    return fig


# |%%--%%| <7SCkZ84APq|GzrZNInjWj>

pcs = torch.load("./results/vae_mnist/samples.pt").cpu()

# Rotate upwards.
phi = torch.tensor(90 * torch.pi / 180)
s = torch.sin(phi)
c = torch.cos(phi)
rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).unsqueeze(0)

pcs = pcs @ rot

fig = plot_pc(pcs[32:], num_pc=10)

plt.savefig("generated_mnist_samples.png")


# |%%--%%| <GzrZNInjWj|WuugzMmmFF>


def plot_recon_2d(recon_pcs, ref_pcs, num_pc=5):

    fig, axes = plt.subplots(nrows=2, ncols=num_pc, figsize=(num_pc * 2, 2 * 2))

    for recon_pc, ref_pc, axis in zip(recon_pcs, ref_pcs, axes.T):
        recon_pc = rotate(recon_pc.reshape(-1, 2), degrees=-90)
        ref_pc = rotate(ref_pc.reshape(-1, 2), degrees=-90)

        ax = axis[0]
        ax.scatter(recon_pc[:, 0], recon_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")

        ax = axis[1]
        ax.scatter(ref_pc[:, 0], ref_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")
    return fig


pcs_refs = torch.load("./results/encoder_mnist/references.pt").cpu()
pcs_recon = torch.load("./results/encoder_mnist/reconstructions.pt").cpu()

fig = plot_recon_2d(pcs_recon, pcs_refs, num_pc=10)

plt.savefig("reconstructed_mnist.png")
