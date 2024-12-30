import re
import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset

from datasets.transforms import CenterTransform, FixedLength, Skeleton
from datasets.base_dataset import BaseModule, BaseConfig
from datasets.transforms import MnistTransform, EctTransform


from dataclasses import dataclass

from layers.ect import EctConfig


@dataclass
class DataModuleConfig(BaseConfig):
    root: str = "./data/mnist-pointcloud-sparse"
    num_pts: int = 16
    ectconfig: EctConfig = EctConfig(
        num_thetas=32,
        resolution=32,
        ambient_dimension=2,
        r=1.1,
        scale=32,
        ect_type="points",
        normalized=True,
        seed=2024,
    )
    module: str = "datasets.mnist_sparse"
    force_reload: bool = False


class DataModule(BaseModule):
    def __init__(self, config, force_reload=False):

        self.force_reload = force_reload or config.force_reload
        self.config = config
        self.transform = transforms.Compose(
            [
                Skeleton(),
                MnistTransform(),
                FixedLength(config.num_pts),
                CenterTransform(),
                EctTransform(self.config.ectconfig),
            ]
        )
        super().__init__()

    def prepare_data(self):
        MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            train=True,
            force_reload=self.force_reload,
        )

    def setup(self, **kwargs):
        self.entire_ds = MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            train=True,
            force_reload=self.force_reload,
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = MnistDataset(
            root=self.config.root,
            pre_transform=self.transform,
            train=False,
            force_reload=self.force_reload,
        )


class MnistDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        train=True,
        pre_filter=None,
        force_reload=False,
    ):
        self.train = train
        self.root = root
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
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
