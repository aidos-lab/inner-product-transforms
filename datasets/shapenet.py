"""
Some notes:

We are _not_ downloading the data, we expect  it to be in the folder
./data/shapenetcore/raw/airplane. 
For preprocessing we only divide 

"""

from typing import Callable, List, Literal, Optional
from torch_geometric import transforms
from dataclasses import dataclass, field
import glob
import os.path as osp
import os
import torch
import numpy as np

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

from datasets.transforms import CenterTransform, RandomRotate
from datasets.base_dataset import BaseModule, BaseConfig


synsetid_to_cate = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


@dataclass
class DataModuleConfig(BaseConfig):
    categories: list = field(default_factory=lambda: ["airplane"])
    root: str = "./data/shapenet"
    module: str = "datasets.shapenet"
    samplepoints: int = 2048


class DataModule(BaseModule):
    def __init__(self, config: DataModuleConfig):
        self.config = config
        self.pre_transform = transforms.Compose([CenterTransform()])
        self.categories = config.categories
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, **kwargs):
        self.entire_ds = ShapeNetCore(
            root=self.config.root,
            pre_transform=self.pre_transform,
            transform=RandomRotate(degrees=360/32,axis=1),
            categories=self.categories,
            split="train",
        )
        self.train_ds = ShapeNetCore(
            root=self.config.root,
            pre_transform=self.pre_transform,
            categories=self.categories,
            split="train",
        )

        self.val_ds = ShapeNetCore(
            root=self.config.root,
            pre_transform=self.pre_transform,
            categories=self.categories,
            split="val",
        )
        self.test_ds = ShapeNetCore(
            root=self.config.root,
            pre_transform=self.pre_transform,
            categories=self.categories,
            split="test",
        )


class ShapeNetCore(InMemoryDataset):

    def __init__(
        self,
        root: str,
        categories: list = field(default_factory=lambda: ["airplane"]),
        split: Literal["train", "test", "val"] = "train",
        transform: Optional[Callable] | None = None,
        pre_transform: Optional[Callable] | None = None,
        pre_filter: Optional[Callable] | None = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split

        self.categories = categories
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        if self.processed_paths:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return list(synsetid_to_cate.keys())

    @property
    def processed_file_names(self) -> List[str]:
        return [
            f"{self.categories[0]}_{self.split}.pt",
        ]

    def process(self) -> None:
        self.save(self.process_set(), self.processed_paths[0])

    def process_set(self) -> List[Data]:
        categories_ids = [cate_to_synsetid[cat] for cat in self.categories]
        data_list = []
        for target, category in enumerate(categories_ids):
            folder = osp.join(self.raw_dir, "ShapeNetCore.v2.PC15k", category)
            paths = glob.glob(f"{folder}/{self.split}/*.npy")

            for path in paths:
                idxs = torch.randperm(15000)[:2048]
                data = Data(
                    x=torch.tensor(np.load(path))[idxs], y=torch.tensor([target])
                )

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return data_list

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.name}({len(self)})"
