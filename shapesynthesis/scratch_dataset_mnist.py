from datasets.mnist import DataModule
from layers.ect import EctConfig
import matplotlib.pyplot as plt
from loaders import load_config

config = load_config("./shapesynthesis/configs/encoder_mnist.yaml")
print(config)

dm = DataModule(config=config.data, force_reload=False)
data1 = dm.test_ds[0]
data2 = dm.test_ds[0]


for batch in dm.test_dataloader():
    print(batch.ect.shape)
    break
# print(batch.ect.min())
# print(batch.ect.max())


plt.imshow(batch.ect[0].cpu().squeeze().numpy())
plt.show()
