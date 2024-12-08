from datasets.shapenetcore import DataModule, DataModuleConfig
import torch
from layers.ect import EctConfig

# config = load_config("./configs/encoder_chair_sparse.yaml")
# config = config.data
# config.force_reload = True
dm = DataModule(
    DataModuleConfig(
        cates=["car"],
        ectconfig=EctConfig(num_thetas=128, bump_steps=128),
        force_reload=False,
    )
)

for test_batch in dm.test_dataloader():
    print(test_batch.ect.norm(dim=-1).max())
    break

for val_batch in dm.val_dataloader():
    print(val_batch.ect.norm(dim=-1).max())
    break


print("len", len(dm.val_ds))

print(torch.allclose(val_batch.ect, test_batch.ect))
print(val_batch.ect.shape)
