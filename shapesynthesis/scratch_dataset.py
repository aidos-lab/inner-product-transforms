from datasets.mnist import DataModule, DataModuleConfig
from layers.ect import EctConfig
import matplotlib.pyplot as plt
from loaders import load_config, load_model
from torch_geometric.sampler import BaseSampler
from torchvision.datasets import MNIST


# config = load_config("./shapesynthesis/configs/encoder_mnist.yaml")


dm = DataModule(
    config=DataModuleConfig(),
    force_reload=True,
)
# data1 = dm.test_ds[0]
# data2 = dm.test_ds[0]

# # batch = Batch.from_data_list([data1, data2])
# # print(batch.ect)


for batch in dm.test_dataloader():
    print(batch[0].num_nodes)
    break


# # print(batch.ect.min())
# # print(batch.ect.max())

idx = 3
plt.imshow(batch[idx].ect.cpu().squeeze().numpy())
plt.show()

x = batch[idx].x.cpu().squeeze().numpy()
plt.scatter(x[:, 0], x[:, 1])
plt.show()
