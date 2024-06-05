import torch
from torch.utils.data import random_split

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from disentanglement_datasets import DSprites
from torch_geometric.data import Data

from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import FaceToEdge

from datasets.transforms import CenterTransform, SkeletonGraph
from datasets.base_dataset import DataModule
from datasets.transforms import DspritesTransform, EctTransform
from datasets.config import DataModuleConfig

from dataclasses import dataclass


@dataclass
class DspritesDataModuleConfig(DataModuleConfig):
    root: str = "./data"
    module: str = "datasets.dsprites"


class DspritesDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        # self.transform = transforms.Compose(
        #     [MnistTransform(), CenterTransform(), EctTransform()]
        # )
        self.transform = transforms.Compose([DspritesTransform()])
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def setup(self):
        self.entire_ds = DspritesDataset(
            root=self.config.root, pre_transform=self.transform
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = DspritesDataset(
            root=self.config.root, pre_transform=self.transform
        )


class DspritesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        DSprites(root=f"{self.root}/raw/", download=True)

    def process(self):
        ds = DSprites(root=f"{self.root}/raw/", download=True)
        ds = [
            Data(
                x=input_dict["input"],
                latent=input_dict["latent"],
            )
            for input_dict in ds
        ]
        if self.pre_transform is not None:

            data_list = [self.pre_transform(data) for data in ds]

        train_data, train_slices = self.collate(data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
