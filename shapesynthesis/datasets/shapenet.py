import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from shapesynthesis.datasets.shapenetbase import get_datasets as get_raw_datasets
from shapesynthesis.layers.ect import EctConfig


@dataclass
class DataConfig:
    module: str = "datasets.shapenet"
    root_dir: str = "./data/shapenet"
    raw_dir: str = "./data/shapenet/raw/ShapeNetCore.v2.PC15k"
    cates: list = field(default_factory=lambda: ["airplane"])
    root: str = "./data/shapenet"
    batch_size: int = 64


####################################
### Utility
####################################


def cates_to_string(cates: list):
    if len(cates) == 1:
        cates_string = cates[0]
    else:
        cates_string = "_".join(cates)
    return cates_string


####################################
### Create Datasets
####################################


class ShapnetDataset(Dataset):
    """
    Dataset interface to add transforms at runtime later.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample[:2048]


def create_dataset(config: DataConfig, dev: bool = False):
    """
    Create the datasets for processing. Creates either
    the dev dataset or the full dataset.
    """

    dataset_type = "dev/" if dev else "prod/"

    # Create the base name for the data folder from the cates used.
    cates_string = cates_to_string(config.cates)

    path = f"{config.root_dir}/{cates_string}/{dataset_type}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tr_ds, vl_ds, te_ds = get_raw_datasets(config.cates, config.raw_dir)

    # For the dev dataset we only use a small subset of the data.
    if dev:
        m = torch.tensor(tr_ds.all_points_mean.reshape(1, -1))
        s = torch.tensor(tr_ds.all_points_std.reshape(1, -1))
        tr_ds = torch.tensor(tr_ds.all_points[:64])
        vl_ds = torch.tensor(vl_ds.all_points[:64])
        te_ds = torch.tensor(te_ds.all_points[:64])
    else:
        m = torch.tensor(tr_ds.all_points_mean.reshape(1, -1))
        s = torch.tensor(tr_ds.all_points_std.reshape(1, -1))
        tr_ds = torch.tensor(tr_ds.all_points)
        vl_ds = torch.tensor(vl_ds.all_points)
        te_ds = torch.tensor(te_ds.all_points)

    # Save the tensors to the correct folders
    torch.save(tr_ds, f"{path}train.pt")
    torch.save(te_ds, f"{path}test.pt")
    torch.save(vl_ds, f"{path}val.pt")
    torch.save(m, f"{path}m.pt")
    torch.save(s, f"{path}s.pt")


def create_all_datasets(config: DataConfig):
    """Creates all datasets. First the dev dataset in case of bugs."""
    create_dataset(config=config, dev=True)
    create_dataset(config=config, dev=False)


def get_dataloader(config: DataConfig, split: str, dev: bool):
    """
    Creates a single dataloader for a given split and for either
    dev or prod.
    """
    dataset_type = "dev" if dev else "prod"
    cates_string = cates_to_string(config.cates)
    dataset_path = f"{config.root_dir}/{cates_string}/{dataset_type}/{split}.pt"
    dataset_mean = f"{config.root_dir}/{cates_string}/{dataset_type}/m.pt"
    dataset_std = f"{config.root_dir}/{cates_string}/{dataset_type}/s.pt"

    ds = torch.load(dataset_path, weights_only=False)
    m = torch.load(dataset_mean, weights_only=False)
    s = torch.load(dataset_std, weights_only=False)

    shuffle = False if split in ["test", "val"] else True
    if dev:
        drop_last = False
    else:
        drop_last = True if split in ["train"] else False

    dataset = ShapnetDataset(ds)

    return (
        DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=0,
            shuffle=shuffle,
            drop_last=drop_last,
        ),
        m,
        s,
    )


def get_all_dataloaders(config: DataConfig, dev: bool):
    train_dl, m, s = get_dataloader(config, "train", dev=dev)
    val_dl, _, _ = get_dataloader(config, "val", dev=dev)
    test_dl, _, _ = get_dataloader(config, "test", dev=dev)

    return train_dl, val_dl, test_dl, m, s


if __name__ == "__main__":
    airplane_config = DataConfig(
        root_dir="./data/shapenet",
        raw_dir="./data/shapenet/raw/ShapeNetCore.v2.PC15k",
        cates=[
            "airplane",
        ],
        batch_size=32,
        module="",
    )
    car_config = DataConfig(
        root_dir="./data/shapenet",
        raw_dir="./data/shapenet/raw/ShapeNetCore.v2.PC15k",
        cates=[
            "car",
        ],
        batch_size=32,
        module="",
    )
    chair_config = DataConfig(
        root_dir="./data/shapenet",
        raw_dir="./data/shapenet/raw/ShapeNetCore.v2.PC15k",
        cates=[
            "chair",
        ],
        batch_size=32,
        module="",
    )
    create_dataset(airplane_config, dev=False)
    create_dataset(airplane_config, dev=True)

    create_dataset(car_config, dev=False)
    create_dataset(car_config, dev=True)

    create_dataset(chair_config, dev=False)
    create_dataset(chair_config, dev=True)
