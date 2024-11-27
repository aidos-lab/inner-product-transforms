from datasets.shapenetcore import DataModule, DataModuleConfig
from layers.ect import EctConfig
from torch_geometric.data import Batch
import matplotlib.pyplot as plt


ectconfig = EctConfig(
    num_features=3, num_thetas=256, bump_steps=256, r=5, normalized=True
)
config = DataModuleConfig(cates=["chair"], ectconfig=ectconfig, force_reload=False)

dm = DataModule(config=config)
data1 = dm.test_ds[0]
data2 = dm.test_ds[0]

# batch = Batch.from_data_list([data1, data2])
# print(batch.ect)


for batch in dm.train_dataloader():
    print(batch)
    break
print(batch.ect.min())
print(batch.ect.max())


plt.imshow(batch.ect[0].cpu().squeeze().numpy())
plt.show()
