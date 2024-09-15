"""
All transforms for the datasets.
"""

from typing import List, Tuple, Union
import math
import random
import torch
import torchvision
import numpy as np

from torch_geometric.utils import degree
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import (
    KNNGraph,
    RandomJitter,
    BaseTransform,
    LinearTransformation,
)

from skimage.morphology import skeletonize

from layers.ect import EctLayer, EctConfig


class RandomScale:
    def __init__(self, scales: Tuple[float, float]) -> None:
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data: Data) -> Data:
        assert data.x is not None

        scale = random.uniform(*self.scales)

        m = data.x.mean(dim=-2)
        x = data.x - m
        n = x.norm(dim=-1, keepdim=True)
        x /= n
        data.x = (x * scale) * n + m
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.scales})"


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
        return Data(x=ect.unsqueeze(0), pts=data.x)


class ClassFilter:
    def __init__(self, classes: List):
        self.classes = classes

    def __call__(self, data: Data) -> bool:
        # Return classes
        if data.y.item() in self.classes:
            return True
        else:
            return False


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
            [
                self.lin[coordinates[0]].squeeze(),
                self.lin[coordinates[1]].squeeze(),
            ]
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

        return Data(
            x=torch.vstack([self.X[idx], self.Y[idx]]).T,
            # face=torch.tensor(dly.cells(), dtype=torch.long).T,
            y=torch.tensor(y, dtype=torch.long),
        )


class DspritesTransform:
    def __init__(self):
        xcoords = torch.linspace(-0.5, 0.5, 64)
        ycoords = torch.linspace(-0.5, 0.5, 64)
        self.X, self.Y = torch.meshgrid(xcoords, ycoords)
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data) -> Data:
        idx = torch.nonzero(data.x, as_tuple=True)

        return Data(
            x=torch.vstack([self.X[idx], self.Y[idx]]).T,
            # face=torch.tensor(dly.cells(), dtype=torch.long).T,
            latent=data.latent,
        )


class FixedLength:
    def __init__(self, length=128):
        self.length = length

    def __call__(self, data):
        res = data.clone()
        if data.x.shape[0] < self.length:
            idx = torch.tensor(np.random.choice(len(data.x), self.length, replace=True))
        else:
            idx = torch.tensor(
                np.random.choice(len(data.x), self.length, replace=False)
            )
        res.x = data.x[idx]
        return res


class FixedLengthBZR:
    def __init__(self, length=128):
        self.length = length

    def __call__(self, data):
        res = data.clone()
        if data.x.shape[0] < self.length:
            idx = torch.tensor(
                np.random.choice(
                    len(data.x), self.length - data.x.shape[0], replace=True
                )
            )
            res.x = torch.vstack([data.x, data.x[idx]])
            return res
        else:
            idx = torch.tensor(
                np.random.choice(len(data.x), self.length, replace=False)
            )
            res.x = data.x[idx]
            return res


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


class RandomRotate(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(
        self,
        degrees: Union[Tuple[float, float], float],
        axis: int = 0,
    ) -> None:
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def forward(self, data: Data) -> Data:
        data.pos = data.x
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.x.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        data = LinearTransformation(torch.tensor(matrix))(data)
        data.x = data.pos
        data.pos = None
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.degrees}, " f"axis={self.axis})"
