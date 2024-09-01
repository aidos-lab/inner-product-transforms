import torch
from omegaconf import OmegaConf

from datasets.shapenetcore import DataModule, DataModuleConfig 

import numpy as np

dm = DataModule(DataModuleConfig(cates=["airplane"]))



for batch in dm.train_dataloader():
    print(len(batch))

