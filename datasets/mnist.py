import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset

from datasets.transforms import CenterTransform, FixedLength, SkeletonGraph
from datasets.base_dataset import BaseModule, BaseConfig
from datasets.transforms import MnistTransform, EctTransform
from torch_geometric.data import Batch, Data, InMemoryDataset, makedirs
from torch_geometric.transforms import BaseTransform, LinearTransformation
from typing import List, Tuple, Union
import math
import random

from dataclasses import dataclass


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


@dataclass
class DataModuleConfig(BaseConfig):
    root: str = "./data/MNIST"
    module: str = "datasets.mnist"


class EctMnistDataModule(BaseModule):
    def __init__(self, config):
        self.config = config
        # self.transform = transforms.Compose(
        #     [MnistTransform(), CenterTransform(), EctTransform()]
        # )
        self.transform = transforms.Compose(
            [SkeletonGraph(), CenterTransform(), EctTransform()]
        )
        super().__init__(
            config.root,
            config.batch_size,
            config.num_workers,
            config.pin_memory,
        )

    def setup(self):
        self.entire_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=True
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=False
        )


class MnistDataModule(BaseModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [MnistTransform(), FixedLength(), CenterTransform()]
        )
        super().__init__(
            config.root,
            config.batch_size,
            config.num_workers,
            config.pin_memory,
        )

    def prepare_data(self):
        MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=True
        )

    def setup(self, **kwargs):
        self.entire_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=True
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=False
        )


class DataModule(BaseModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [MnistTransform(), FixedLength(), CenterTransform()]
        )
        super().__init__(
            config.root,
            config.batch_size,
            config.num_workers,
            config.pin_memory,
        )

    def prepare_data(self):
        MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            train=True,
        )

    def setup(self, **kwargs):
        self.entire_ds = MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            transform=RandomRotate(degrees=90),
            train=True,
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=False
        )


class MnistDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        train=True,
        pre_filter=None,
    ):
        self.train = train
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["MNIST"]

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt"]

    def download(self):
        if self.train:
            MNIST(f"{self.root}/raw/", train=True, download=True)
        else:
            MNIST(f"{self.root}/raw/", train=False, download=True)

    def process(self):
        train_ds = MNIST(f"{self.root}/raw/", train=True, download=True)
        test_ds = MNIST(f"{self.root}/raw/", train=False, download=True)

        if self.pre_transform is not None:
            train_data_list = [self.pre_transform(data) for data in train_ds]
            test_data_list = [self.pre_transform(data) for data in test_ds]

        train_data, train_slices = self.collate(train_data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
        test_data, test_slices = self.collate(test_data_list)
        torch.save((test_data, test_slices), self.processed_paths[1])
