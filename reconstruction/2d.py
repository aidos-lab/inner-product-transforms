from networkx import reconstruct_path
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric
from datasets.tu import TULetterLowConfig, TUDataModule
from skimage.transform import iradon
from torch_geometric.data import Data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

np.random.seed(42)

NUM_STEPS = 512
NUM_PTS = 15
SCALE = 200


def compute_ect(x, v, ei=None):
    nh = x @ v
    lin = torch.linspace(-1.3, 1.3, NUM_STEPS).view(-1, 1, 1)
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


def fbp(ect):
    sinogram = ect.numpy()
    theta = torch.linspace(0, 360, NUM_STEPS).numpy()
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name="ramp") * 100
    return reconstruction_fbp


def get_node_coordinates(reconstruction_fbp):
    im = img_as_float(reconstruction_fbp)
    coordinates = peak_local_max(im, min_distance=20, threshold_rel=0.1)
    lin = torch.linspace(-1.3, 1.3, NUM_STEPS).view(-1, 1, 1)
    x_hat = torch.vstack(
        [lin[512 - coordinates[:, 0]].squeeze(), lin[coordinates[:, 1]].squeeze()]
    ).T
    return x_hat


def get_edge_indices(reconstruction_fbp, x_hat):
    # Reconstruct adjacency matrix
    recon_ect = []
    for x_hat_i in x_hat:
        recon_ect.append(compute_ect(x_hat_i, v))

    adj = torch.zeros((len(x_hat), len(x_hat)))
    theta = torch.linspace(0, 360, NUM_STEPS).numpy()
    for i in range(len(x_hat)):
        for j in range(len(x_hat)):
            ect = compute_ect((x_hat[i] + x_hat[j]) / 2, v)
            rec = torch.tensor(
                iradon(ect.numpy(), theta=theta, filter_name="ramp"), dtype=torch.float
            )

            adj[i, j] = -1 * (rec * reconstruction_fbp).sum()

    adj[adj < 1] = 0
    adj[adj > 1] = 1
    return torch.nonzero(adj).T


def reconstruct_ect(ect):
    reconstruction_fbp = fbp(ect)
    x_hat = get_node_coordinates(reconstruction_fbp)
    edge_index_recon = get_edge_indices(reconstruction_fbp, x_hat)
    return x_hat, edge_index_recon


if __name__ == "__main__":
    dataset = TUDataModule(TULetterLowConfig())
    data = dataset.entire_ds[8]
    x = data.x
    ei = data.edge_index
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(g, pos=x.numpy())
    plt.show()

    v = torch.vstack(
        [
            torch.sin(torch.linspace(0, 2 * torch.pi, NUM_STEPS)),
            torch.cos(torch.linspace(0, 2 * torch.pi, NUM_STEPS)),
        ]
    )

    ect = compute_ect(x, v, ei=ei)
    x_hat, edge_index_recon = reconstruct_ect(ect)

    data_recon = Data(x=x_hat, edge_index=edge_index_recon)
    g = torch_geometric.utils.to_networkx(data_recon, to_undirected=True)
    nx.draw(g, pos=data_recon.x.numpy())
    plt.show()
