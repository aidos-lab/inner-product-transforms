import torch
from omegaconf import OmegaConf

from datasets.shapenetcore import DataModule, DataModuleConfig 

import numpy as np

dm = DataModule(DataModuleConfig(cates=["chair"]))



for batch in dm.test_dataloader():
    print(batch.x.norm(dim=-1).max())

