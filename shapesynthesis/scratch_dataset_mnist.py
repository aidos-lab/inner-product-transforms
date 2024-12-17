from datasets.mnist import DataModule, DataModuleConfig
from layers.ect import EctConfig
import matplotlib.pyplot as plt
from loaders import load_config


dm = DataModule(
    DataModuleConfig(
        root="./data/mnistpointcloud",
        ectconfig=EctConfig(
            num_thetas=32,
            r=1.1,
            scale=16,
            ect_type="points",
            resolution=32,
            ambient_dimension=2,
            normalized=True,
            seed=2024,
        ),
        batch_size=64,
        num_pts=128,
    ),
    force_reload=True,
)

data1 = dm.test_ds[0]
data2 = dm.test_ds[0]


for batch in dm.test_dataloader():
    print(batch.ect.shape)
    break
# print(batch.ect.min())
# print(batch.ect.max())

print(batch.ect.shape)
import torch

torch.save(batch.ect, "ectbatch.pt")

plt.imshow(batch.ect[0].cpu().squeeze().numpy())
plt.show()
