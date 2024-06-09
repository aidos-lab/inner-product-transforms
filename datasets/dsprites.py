import torch
from torch.utils.data import random_split

import torchvision.transforms as transforms
from disentanglement_datasets import DSprites
from torch_geometric.data import Data, InMemoryDataset

from datasets.base_dataset import DataModule, DataModuleConfig
from datasets.transforms import CenterTransform, DspritesTransform


from dataclasses import dataclass

from datasets.transforms import FixedLength


@dataclass
class DspritesDataModuleConfig(DataModuleConfig):
    root: str = "./data/dsprites"
    module: str = "datasets.dsprites"


class DspritesDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [DspritesTransform(), FixedLength(), CenterTransform()]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def prepare_data(self):
        DspritesDataset(root=self.config.root, pre_transform=self.transform)

    def setup(self, **kwargs):
        self.entire_ds = DspritesDataset(
            root=self.config.root, pre_transform=self.transform
        )
        self.train_ds, val_ds = random_split(
            self.entire_ds,
            [
                int(0.1 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.1 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.val_ds, _ = random_split(
            val_ds,
            [
                int(0.1 * len(val_ds)),
                len(val_ds) - int(0.1 * len(val_ds)),
            ],
        )  # type: ignore

        # self.test_ds = DspritesDataset(
        #     root=self.config.root, pre_transform=self.transform
        # )


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
