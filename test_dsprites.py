# from disentanglement_datasets import DSprites
import matplotlib.pyplot as plt


# dataset = DSprites(root="./data", download=True)
# data = dataset[0]
# plt.imshow(data["input"])
# plt.show()

from datasets.dsprites import DspritesDataModule, DspritesDataModuleConfig


dataset = DspritesDataModule(DspritesDataModuleConfig)


data = dataset.entire_ds[0]
plt.scatter(data.x[:, 0], data.x[:, 1])
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
