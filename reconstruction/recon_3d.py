from dataclasses import dataclass
from networkx import center
import torch
import torch.nn as nn
import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import torchvision.transforms.functional as f
from torch_geometric.utils import erdos_renyi_graph
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter, minimum_filter

# np.random.seed(33)

NUM_STEPS = 128
NUM_PTS = 5
SCALE = 300
lin = np.linspace(-1, 1, NUM_STEPS, endpoint=False)


def generate_thetas():
    v = []
    for theta in torch.linspace(0, torch.pi, 8):
        for phi in torch.linspace(0, torch.pi, 64):
            v.append(
                torch.tensor(
                    [
                        torch.sin(phi) * torch.cos(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(phi),
                    ]
                )
            )
    return torch.vstack(v).T


def compute_ect(x, v, ei=None):
    nh = x @ v
    lin = torch.linspace(-1, 1, NUM_STEPS).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(SCALE * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(SCALE * torch.sub(lin, nh))
    )
    ecc = ecc.sum(axis=1)
    if ei is not None:
        eh = nh[ei].mean(axis=0)
        eccedge = torch.nn.functional.sigmoid(SCALE * torch.sub(lin, eh)) * (
            1 - torch.nn.functional.sigmoid(SCALE * torch.sub(lin, eh))
        )
        eccedge = eccedge.sum(axis=1)
        ecc -= eccedge
    return ecc


################################################################################
### Filtered backprojection
################################################################################
def fpb(ect, v):
    xg, yg, zg = np.meshgrid(
        np.linspace(-1, 1, NUM_STEPS, endpoint=False),
        np.linspace(-1, 1, NUM_STEPS, endpoint=False),
        np.linspace(-1, 1, NUM_STEPS, endpoint=False),
        indexing="ij",
        sparse=True,
    )

    recon = torch.zeros(NUM_STEPS, NUM_STEPS, NUM_STEPS)

    def calc_idx(theta, xg, yg, zg):
        heights = theta[0] * xg + theta[1] * yg + theta[2] * zg
        idx = ((heights + 1) * NUM_STEPS / 2).long() + 1
        idx[idx > NUM_STEPS - 1] = NUM_STEPS - 1
        return idx

    i = 0
    for theta, slice in zip(v.T, ect.T):
        i += 1
        idx = calc_idx(theta, xg, yg, zg)
        reps = slice[idx]
        recon += reps
    return recon


################################################################################
### Recover node coordinates
################################################################################


def get_node_coordinate(recon_np):

    res = maximum_filter(recon_np, footprint=np.ones((11, 11, 11)))
    mask = recon_np == res
    idxx, idxy, idxz = np.nonzero(mask)
    vals = recon_np[idxx, idxy, idxz]
    vals /= vals.max()
    idx = np.where(vals > 0.7)
    idx_x = idxx[idx]
    idx_y = idxy[idx]
    idx_z = idxz[idx]
    return torch.tensor(np.vstack([lin[idx_x], lin[idx_y], lin[idx_z]]).T)


################################################################################
### Recover edge coordinates
################################################################################


def get_edge_coordinates(recon_np):
    res = minimum_filter(recon_np, footprint=np.ones((11, 11, 11)))
    mask = recon_np == res

    idxx, idxy, idxz = np.nonzero(mask)
    vals = recon_np[idxx, idxy, idxz]
    vals /= vals.min()
    idx = np.where(vals > 0.7)

    idx_x = idxx[idx]
    idx_y = idxy[idx]
    idx_z = idxz[idx]

    lin = np.linspace(-1, 1, NUM_STEPS, endpoint=False)

    return torch.tensor(np.vstack([lin[idx_x], lin[idx_y], lin[idx_z]]).T)


################################################################################
### Calculate edge indices
################################################################################


def get_edge_indices(pts, edge_pts):
    ei_recon = []
    for i in range(len(pts)):
        for j in range(len(pts)):
            pt_i = pts[i].reshape(1, 3)
            pt_j = pts[j].reshape(1, 3)
            pt = (pt_i + pt_j) / 2

            for epts in edge_pts:
                if torch.norm(epts - pt) < 0.2:
                    ei_recon.append([i, j])
    return ei_recon


def center_points(x):
    x -= x.mean(axis=0)
    x /= x.pow(2).sum(axis=1).sqrt().max()
    return x


def main(ect, v):
    recon = fpb(ect, v)
    pts = get_node_coordinate(recon_np=recon.numpy())
    recon_np = recon.numpy()
    edge_pts = get_edge_coordinates(recon_np)
    ei_recon = get_edge_indices(pts, edge_pts)
    ei_unique = list(set([tuple(sorted(l)) for l in ei_recon]))
    return pts, torch.tensor(ei_unique).T, edge_pts


if __name__ == "__main__":

    # Generate data
    v = generate_thetas()
    x = torch.tensor(
        np.random.uniform(-0.7, 0.7, size=(NUM_PTS, 3)), dtype=torch.float
    )
    x[:, 2] *= 0
    x = center_points(x)

    ei = torch.tensor([[0, 1, 2], [1, 2, 4]])

    ect = compute_ect(x, v, ei=ei)
    pts, ei_recon, edge_pts = main(ect, v)
    # print(pts.shape)
    # print(pts)

    ei_true = []
    for ei_idx in ei.T:
        ei_true.append((x[ei_idx[0]] + x[ei_idx[1]]) / 2)

    ei_true = torch.vstack(ei_true)

    print("Predicted numper of points\n", pts)
    plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(ei_true[:, 0], ei_true[:, 1])
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.scatter(edge_pts[:, 0], edge_pts[:, 1])

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

    print(ei_recon)
    # print(edge_pts)

    # plt.scatter(pts[:, 0], pts[:, 1])
    # plt.scatter(edge_pts[:, 0], edge_pts[:, 1])

    # plt.show()
