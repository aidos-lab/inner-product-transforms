from torch_geometric.datasets import SHREC2016
from torch_geometric import transforms
from datasets.base_dataset import DataModule
from torch.utils.data import random_split

from torch_geometric.data import Batch, Data
from datasets.config import DataModuleConfig

from datasets.transforms import CenterTransform, PosToXTransform
from dataclasses import dataclass


@dataclass
class ShrecDataModuleConfig(DataModuleConfig):
    root: str = "./data/shrec"
    module: str = "datasets.shrec"
    samplepoints: int = 500


class ShrecDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                # transforms.SamplePoints(self.config.samplepoints),
                PosToXTransform(),
                CenterTransform(),
            ]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    
    def setup(self):
        self.entire_ds = SHREC2016(
            root=self.config.root,
            pre_transform=self.pre_transform,
            partiality="Holes",
            category="Centaur",
            train=True,
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.8 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.8 * len(self.entire_ds)),
            ],
        )  # type: ignore
        self.test_ds = SHREC2016(
            root=self.config.root,
            pre_transform=self.pre_transform,
            partiality="Holes",
            category="Centaur",
            train=False,
        )