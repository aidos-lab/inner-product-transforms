from datasets.mnist import DataModule
from layers.ect import EctConfig
import matplotlib.pyplot as plt
from loaders import load_config

ectconfig = EctConfig(
    num_features=3, num_thetas=256, bump_steps=256, r=5, normalized=True
)
# config = DataModuleConfig(root="./data/mnistpointcloud")
config = load_config("./configs/encoder_mnist.yaml")


dm = DataModule(config=config.data, force_reload=False)
data1 = dm.test_ds[0]
data2 = dm.test_ds[0]

# batch = Batch.from_data_list([data1, data2])
# print(batch.ect)


for batch in dm.test_dataloader():
    print(batch[0].num_nodes)
    break
# print(batch.ect.min())
# print(batch.ect.max())


# plt.imshow(batch.ect[0].cpu().squeeze().numpy())
# plt.show()
