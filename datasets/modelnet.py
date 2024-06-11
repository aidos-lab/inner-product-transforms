from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from torch.utils.data import random_split
from dataclasses import dataclass

from datasets.transforms import CenterTransform, ClassFilter, ModelNetTransform
from datasets.base_dataset import BaseModule, BaseConfig


class CenterTransformNew(object):
    def __call__(self, data):
        data.x -= data.x.mean(axis=0)
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


@dataclass
class DataModuleConfig(BaseConfig):
    root: str = "./data/modelnet"
    name: str = "40"
    module: str = "datasets.modelnet"
    samplepoints: int = 100


class DataModule(BaseModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                transforms.SamplePoints(self.config.samplepoints),
                ModelNetTransform(),
                CenterTransform(),
            ]
        )
        self.pre_filter = ClassFilter()
        super().__init__(
            config.root,
            config.batch_size,
            config.num_workers,
            config.pin_memory,
        )

    def prepare_data(self):
        pass

    def setup(self, **kwargs):
        self.entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
            pre_filter=self.pre_filter,
            name=self.config.name,
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.8 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.8 * len(self.entire_ds)),
            ],
        )  # type: ignore
        self.test_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            train=False,
            name=self.config.name,
        )
