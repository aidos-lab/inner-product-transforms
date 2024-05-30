import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torchvision
import vedo
from torch_geometric.transforms import KNNGraph, RandomJitter
from layers.ect import EctLayer, EctConfig
from torch_geometric.data import Batch, Data
from skimage.morphology import skeletonize
import numpy as np


def plot_batch(data):
    coords = data.x.cpu().numpy()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    sequence_containing_x_vals = coords[:, 0]
    sequence_containing_y_vals = coords[:, 1]
    sequence_containing_z_vals = coords[:, 2]

    ax.scatter(
        sequence_containing_x_vals,
        sequence_containing_y_vals,
        sequence_containing_z_vals,
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()


class Skeleton:
    def __init__(self):
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data):
        image, y = data
        # perform skeletonization
        skeleton = skeletonize(image.numpy()) / 255
        return (skeleton, y)


class ThresholdTransform(object):
    def __call__(self, data):
        data.x = torch.hstack([data.pos, data.x])
        return data


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean(axis=0)
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


class Normalize(object):
    def __call__(self, data):
        mean = data.x.mean()
        std = data.x.std()
        data.x = (data.x - mean) / std
        return data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class NCI109Transform(object):
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float).unsqueeze(0).T
        atom_number = torch.argmax(data.x, dim=-1, keepdim=True)
        data.x = torch.hstack([deg, atom_number])
        return data


class ModelNetTransform(object):
    def __call__(self, data):
        data.x = data.pos
        data.pos = None
        return data


class PosToXTransform(object):
    def __call__(self, data):
        data.x = data.pos
        data.pos = None
        return data


class Project(object):
    def __call__(self, batch):
        batch.x = batch.x[:, :2]
        # scaling
        return batch


class SkeletonGraph:
    def __init__(self):
        self.tr = torchvision.transforms.ToTensor()
        self.knn = KNNGraph(k=2, force_undirected=True)
        self.lin = torch.linspace(-1, 1, 28)
        self.translate = RandomJitter(translate=0.05)

    def __call__(self, data):
        image, y = data
        # perform skeletonization
        skeleton = skeletonize(self.tr(image).squeeze().numpy())
        coordinates = np.where(skeleton > 0.5)
        x_hat = torch.vstack(
            [self.lin[coordinates[0]].squeeze(), self.lin[coordinates[1]].squeeze()]
        ).T

        data = Data(pos=x_hat, y=y)
        data = self.knn(data)
        data = self.translate(data)
        data.x = data.pos
        data.pos = None
        return data


class MnistTransform:
    def __init__(self):
        xcoords = torch.linspace(-0.5, 0.5, 28)
        ycoords = torch.linspace(-0.5, 0.5, 28)
        self.X, self.Y = torch.meshgrid(xcoords, ycoords)
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data: tuple) -> Data:
        img, y = data
        img = self.tr(img)
        idx = torch.nonzero(img.squeeze(), as_tuple=True)
        # gp = torch.vstack([self.X[idx], self.Y[idx]]).T
        # dly = vedo.delaunay2d(gp, mode="xy", alpha=0.4).c("w").lc("o").lw(1)

        return Data(
            x=torch.vstack([self.X[idx], self.Y[idx]]).T,
            # face=torch.tensor(dly.cells(), dtype=torch.long).T,
            y=torch.tensor(y, dtype=torch.long),
        )


class Mnist3DTransform:
    def __init__(self):
        xcoords = torch.linspace(-0.5, 0.5, 28)
        ycoords = torch.linspace(-0.5, 0.5, 28)
        self.X, self.Y = torch.meshgrid(xcoords, ycoords)
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data: tuple) -> Data:
        img, y = data
        img = self.tr(img)
        idx = torch.nonzero(img.squeeze(), as_tuple=True)
        gp = torch.vstack([self.X[idx], self.Y[idx]]).T
        dly = vedo.delaunay2d(gp, mode="xy", alpha=0.03).c("w").lc("o").lw(1)
        print(dly.faces())
        return Data(
            x=torch.tensor(dly.points()),
            face=torch.tensor(dly.faces(), dtype=torch.long).T,
            y=torch.tensor(y, dtype=torch.long),
        )


class EctTransform:
    def __init__(self, normalized=True):
        config = EctConfig(num_thetas=64, bump_steps=64)
        v = torch.vstack(
            [
                torch.sin(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                torch.cos(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
            ]
        )
        self.layer = EctLayer(config=config, v=v)
        self.normalized = normalized

    def __call__(self, data):
        batch = Batch.from_data_list([data])
        ect = self.layer(batch)
        if self.normalized:
            return Data(x=ect.unsqueeze(0) / ect.max(), pts=data.x)
        else:
            return Data(x=ect.unsqueeze(0), pts=data.x)


class WeightedMnistTransform:
    def __init__(self):
        self.grid = torch.load("./datasets/grid.pt")
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data: tuple) -> Data:
        img, y = data
        img = self.tr(img)

        return Data(
            x=self.grid.x,
            node_weights=img.view(-1, 1),
            edge_index=self.grid.edge_index,
            face=self.grid.face,
            y=torch.tensor(y, dtype=torch.long),
        )
