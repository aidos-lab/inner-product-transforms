# from disentanglement_datasets import DSprites
import matplotlib.pyplot as plt

import torch

# dataset = DSprites(root="./data", download=True)
# data = dataset[0]
# plt.imshow(data["input"])
# plt.show()

from datasets.dsprites import DspritesDataModule, DspritesDataModuleConfig
import numpy as np

dm = DspritesDataModule(DspritesDataModuleConfig())

for batch in dm.train_dataloader():
    break

from torch_geometric.data import Data


pts = batch[0].x
for idx in range(20):
    print(batch[idx].x.shape)

plt.scatter(pts[:, 0], pts[:, 1])
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
