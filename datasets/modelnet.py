from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule
from torch.utils.data import random_split
import torch

from torch_geometric.data import Batch, Data

from torch_geometric.transforms import FaceToEdge
from datasets.transforms import CenterTransform, ModelNetTransform
from layers.config import EctConfig
from layers.ect import EctLayer

class EctTransform:
    def __init__(self, normalized=True):
        self.layer = EctLayer(EctConfig(bump_steps=64,num_thetas=64,num_features=3),fixed=False)
        self.normalized = True
    def __call__(self,data):
        batch = Batch.from_data_list([data])
        ect = self.layer(batch)
        if self.normalized:
            return Data(x=ect/ect.max(),pts=data.x)
        else: 
            return Data(x=ect,pts=data.x)

class ModelNetDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                transforms.SamplePoints(self.config.samplepoints),
                ModelNetTransform(),
                CenterTransform(),
            ]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    
    def setup(self):
        self.entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
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
            train=False,
            name=self.config.name,
        )