"""
All transforms for the datasets.
"""

import math
import random
from typing import Tuple, Union

import numpy as np
import torch
import torchvision
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import (
    BaseTransform,
    KNNGraph,
    LinearTransformation,
    RandomJitter,
)

from shapesynthesis.layers.directions import generate_uniform_directions
from shapesynthesis.layers.ect import EctConfig, EctLayer


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


class EctTransform:
    def __init__(self, ectconfig: EctConfig, normalized=True):
        v = generate_uniform_directions(
            num_thetas=ectconfig.num_thetas,
            d=ectconfig.ambient_dimension,
            seed=ectconfig.seed,
        )
        self.layer = EctLayer(config=ectconfig, v=v)
        self.normalized = normalized

    def __call__(self, data):
        batch = Batch.from_data_list([data])
        ect = self.layer(batch, batch.batch)
        if self.normalized:
            data.ect = (ect / ect.max()).cpu()
            return data

        data.ect = ect.cpu()
        return data


class Normalize(object):
    def __call__(self, data):
        mean = data.x.mean()
        std = data.x.std()
        data.x = (data.x - mean) / std
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


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean(axis=0)
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        data.mean = torch.tensor([0, 0, 0.0])
        data.std = torch.tensor([0.0])
        return data


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
