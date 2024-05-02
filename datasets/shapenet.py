from torch_geometric.datasets import ShapeNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule
from torch.utils.data import random_split

from torch_geometric.data import Batch, Data

from datasets.transforms import CenterTransform, ModelNetTransform
from layers.config import EctConfig
from layers.ect import EctLayer

# CAT = ["Airplane", "Earphone", "Chair","Cap","Guitar", "Knife", "Lamp", "Laptop"]
CAT = ["Airplane", "Earphone"]

class ShapeNetTransform(object):
    def __call__(self, data):
        data.x = data.pos
        data.y = data.category
        data.category = None
        data.pos = None
        return data

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

class ShapeNetDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                # transforms.SamplePoints(self.config.samplepoints),
                ShapeNetTransform(),
                CenterTransform(),
            ]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    
    def setup(self):
        self.entire_ds = ShapeNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            split="train",
            # categories = ["Airplane", "Earphone","Cap", "Chair"],
            categories = CAT,
        )

        self.train_ds = ShapeNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            split="train",
            categories = CAT,
        )

        self.val_ds = ShapeNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            split="val",
            categories = CAT,
        )

        self.test_ds = ShapeNet(
            root=self.config.root,            
            categories = CAT,
            pre_transform=self.pre_transform,
            split="test",
        )
