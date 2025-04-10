import torch

from shapesynthesis.datasets.shapenetcore import DataModule, DataModuleConfig
from shapesynthesis.layers.ect import EctConfig

config = DataModuleConfig(
    ectconfig=EctConfig(
        num_thetas=128,
        resolution=128,
        ambient_dimension=3,
        r=7,
        scale=32,
        ect_type="points",
        normalized=True,
        seed=2024,
    ),
    cates=["airplane"],
    root="./data/shapenet",
    module="shapesynthesis.datasets.shapenetcore",
    force_reload=False,
    num_pts=2048,
)

dm = DataModule(config=config)

torch.save(dm.train_ds.ect, "airplane_ect_128.pt")
