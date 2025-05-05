"""
Base module for the dataset.
"""

from abc import abstractmethod
from dataclasses import dataclass

from lightning import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


@dataclass
class BaseConfig:
    module: str
    root: str = "./data"
    num_workers: int = 0
    batch_size: int = 64
    pin_memory: bool = True
    drop_last: bool = False


class BaseModule(LightningDataModule):
    train_ds: Dataset
    test_ds: Dataset
    val_ds: Dataset
    entire_ds: Dataset
    config: BaseConfig

    def __init__(self):
        super().__init__()

    @abstractmethod
    def setup(self):
        raise NotImplementedError()

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            # sampler=ImbalancedSampler(self.train_ds),
            shuffle=True,
            pin_memory=self.config.pin_memory,
            # drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            # sampler=ImbalancedSampler(self.val_ds),
            shuffle=False,
            pin_memory=self.config.pin_memory,
            # drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            # sampler=ImbalancedSampler(self.test_ds),
            shuffle=False,
            pin_memory=self.config.pin_memory,
            # drop_last=self.drop_last,
        )
