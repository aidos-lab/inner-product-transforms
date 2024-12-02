from datasets.shapenetcore import DataModule, DataModuleConfig
import torch
from loaders import load_config

config = load_config("./configs/encoder_chair_sparse.yaml")
config = config.data
config.force_reload = True
dm = DataModule(config)

for test_batch in dm.test_dataloader():
    print(test_batch.ect.norm(dim=-1).max())
    break

for val_batch in dm.val_dataloader():
    print(val_batch.ect.norm(dim=-1).max())
    break


print(torch.allclose(val_batch.ect, test_batch.ect))
print(val_batch.ect.shape)
