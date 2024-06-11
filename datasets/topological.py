import math
import random
from typing import List, Tuple, Union
from cv2 import NONE_POLISHER
import torchvision.transforms as transforms
from torch_geometric.data import Batch, Data, InMemoryDataset, makedirs
from torch_geometric.transforms import BaseTransform, LinearTransformation

import torch

from layers.ect import EctLayer, EctConfig
from datasets.base_dataset import DataModule, DataModuleConfig
from datasets.transforms import CenterTransform, EctTransform
from dataclasses import dataclass
import numpy as np


@dataclass
class TopologicalDataModuleConfig(DataModuleConfig):
    module: str = "datasets.topological"
    n_samples: int = 1024
    n_manifolds: int = 1000


def sample_from_sphere(n=100, r=1, noise=None, seed=None):
    """Sample `n` data points from a `d`-sphere in `d + 1` dimensions.

    Parameters
    -----------
    n : int
        Number of data points in shape.

    d : int
        Dimension of the sphere.

    r : float
        Radius of sphere.

    noise : float or None
        Optional noise factor. If set, data coordinates will be
        perturbed by a standard normal distribution, scaled by
        `noise`.

    ambient : int or None
        Embed the sphere into a space with ambient dimension equal to
        `ambient`. The sphere is randomly rotated into this
        high-dimensional space.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor
        Tensor of sampled coordinates. If `ambient` is set, array will be
        of shape `(n, ambient)`. Else, array will be of shape `(n, d + 1)`.

    Notes
    -----
    This function was originally authored by Nathaniel Saul as part of
    the `tadasets` package. [tadasets]_

    References
    ----------
    .. [tadasets] https://github.com/scikit-tda/tadasets

    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 3))

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * rng.standard_normal(data.shape)

    return torch.as_tensor(data, dtype=torch.float)


def sample_from_torus(
    n, d=3, r=1.0, R=2.0, noise: float | None = None, seed=None
):
    """Sample points uniformly from torus and embed it in `d` dimensions.

    Parameters
    ----------
    n : int
        Number of points to sample

    d : int
        Number of dimensions.

    r : float
        Radius of the 'tube' of the torus.

    R : float
        Radius of the torus, i.e. the distance from the centre of the
        'tube' to the centre of the torus.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, d)`
        Tensor of sampled coordinates.
    """
    rng = np.random.default_rng(seed)
    angles = []

    while len(angles) < n:
        x = rng.uniform(0, 2 * np.pi)
        y = rng.uniform(0, 1 / np.pi)

        f = (1.0 + (r / R) * np.cos(x)) / (2 * np.pi)

        if y < f:
            psi = rng.uniform(0, 2 * np.pi)
            angles.append((x, psi))

    X = []

    for theta, psi in angles:
        a = R + r * np.cos(theta)
        x = a * np.cos(psi)
        y = a * np.sin(psi)
        z = r * np.sin(theta)

        X.append((x, y, z))

    data = np.asarray(X)

    if noise:
        data += noise * rng.standard_normal(data.shape)

    return torch.as_tensor(data, dtype=torch.float)


def sample_from_unit_cube(n, d=3, noise: float | None = None, seed=None):
    """Sample points uniformly from unit sphere in `d` dimensions with
    respect to the max-norm.


    Parameters
    ----------
    n : int
        Number of points to sample

    d : int
        Number of dimensions.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    torch.tensor of shape `(n, d)`
        Tensor containing the sampled coordinates.
    """
    rng = np.random.default_rng(seed)
    data = rng.uniform(low=-1, high=1, size=(n, d))
    data /= np.max(abs(data), axis=1).reshape(-1, 1)
    if noise:
        data += noise * rng.standard_normal(data.shape)

    return torch.as_tensor(data, dtype=torch.float)


def sample_from_mobius(n: int, noise: float | None = None, seed=NONE_POLISHER):

    a = np.random.uniform(0, 2.0 * np.pi, size=(n,))
    b = np.random.uniform(-0.5, 0.5, size=(n,))

    # This is the Mobius mapping, taking a, b pair and returning an x, y, z
    x = (1 + 0.5 * b * np.cos(a / 2.0)) * np.cos(a)
    y = (1 + 0.5 * b * np.cos(a / 2.0)) * np.sin(a)
    z = 0.5 * b * np.sin(a / 2.0)
    data = np.vstack([x, y, z]).T
    rng = np.random.default_rng(seed)
    if noise:
        data += noise * rng.standard_normal(data.shape)

    return torch.tensor(data, dtype=torch.float)


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
        return (
            f"{self.__class__.__name__}({self.degrees}, " f"axis={self.axis})"
        )


class TopolocigalDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [
                CenterTransform(),
                RandomRotate(degrees=90, axis=0),
                RandomRotate(degrees=90, axis=1),
                RandomRotate(degrees=90, axis=2),
            ]
        )
        super().__init__(
            config.root,
            config.batch_size,
            config.num_workers,
            config.pin_memory,
        )

    def prepare_data(self) -> None:
        TopologicalDataset(
            self.config, split="train", pre_transform=self.transform
        )
        TopologicalDataset(
            self.config, split="test", pre_transform=self.transform
        )
        TopologicalDataset(
            self.config, split="val", pre_transform=self.transform
        )

    def setup(self, **kwargs):
        self.train_ds = TopologicalDataset(
            self.config, split="train", pre_transform=self.transform
        )
        self.test_ds = TopologicalDataset(
            self.config, split="test", pre_transform=self.transform
        )
        self.val_ds = TopologicalDataset(
            self.config, split="val", pre_transform=self.transform
        )
        self.entire_ds = TopologicalDataset(
            self.config, split="train", pre_transform=self.transform
        )


class TopologicalDataset(InMemoryDataset):
    """Represents a 3 segmentation dataset.

    Input params:
        configuration: Configuration dictionary.
    """

    def __init__(
        self, config: TopologicalDataModuleConfig, split, pre_transform
    ):
        self.config = config
        self.split = split
        root = config.root + "/topological"
        super().__init__(root, pre_transform, force_reload=False)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f"raw_{self.split}.pt", f"raw_{self.split}_labels.pt"]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.split}.pt"]

    def download(self) -> None:
        spheres = torch.stack(
            [
                sample_from_sphere(self.config.n_samples, noise=None)
                for i in range(self.config.n_manifolds)
            ]
        )

        cubes = torch.stack(
            [
                sample_from_unit_cube(self.config.n_samples, noise=None)
                for i in range(self.config.n_manifolds)
            ]
        )

        tori = torch.stack(
            [
                sample_from_torus(self.config.n_samples, noise=None)
                for i in range(self.config.n_manifolds)
            ]
        )

        mobius = torch.stack(
            [
                sample_from_mobius(self.config.n_samples, noise=None)
                for i in range(self.config.n_manifolds)
            ]
        )

        self.labels = torch.as_tensor(
            [[0]] * self.config.n_manifolds
            + [[1]] * self.config.n_manifolds
            + [[2]] * self.config.n_manifolds
            + [[3]] * self.config.n_manifolds,
            dtype=torch.long,
        )

        self.data = torch.vstack((spheres, tori, cubes, mobius))

        makedirs(self.root + "/raw")
        torch.save(self.data, self.root + "/raw/" + self.raw_file_names[0])
        torch.save(self.labels, self.root + "/raw/" + self.raw_file_names[1])

    def process(self):
        data_list = []

        manifolds = torch.load(self.root + "/raw/" + self.raw_file_names[0])
        labels = torch.load(self.root + "/raw/" + self.raw_file_names[1])
        for manifold, label in zip(manifolds, labels):
            data_list.append(Data(x=manifold, y=label))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        train_data, train_slices = self.collate(data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
