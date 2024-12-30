"""
Rotated example for mnist.
"""

from dataclasses import dataclass
import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch_geometric.data import InMemoryDataset

from datasets.transforms import (
    CenterTransform,
    FixedLength,
    RandomRotate,
)
from datasets.base_dataset import BaseModule, BaseConfig
from datasets.transforms import MnistTransform


@dataclass
class DataModuleConfig(BaseConfig):
    root: str = "./data/MNIST"
    module: str = "datasets.mnist"


class DataModule(BaseModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [MnistTransform(), FixedLength(), CenterTransform()]
        )
        super().__init__()

    def prepare_data(self):
        MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            train=True,
        )

    def setup(self, *args, **kwargs):
        self.entire_ds = MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            transform=RandomRotate(degrees=180),
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

        train_data_list = []
        test_data_list = []

        if self.pre_transform is not None:
            train_data_list = [self.pre_transform(data) for data in train_ds]
            test_data_list = [self.pre_transform(data) for data in test_ds]

        train_data, train_slices = self.collate(train_data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
        test_data, test_slices = self.collate(test_data_list)
        torch.save((test_data, test_slices), self.processed_paths[1])
