import torch

from shapesynthesis.loaders import load_config, load_datamodule

# Hardcoding the configs for now.
# Run this from the project root.
# Only using the 256**2 ect
# TODO: Adapt the test scripts so that
# add this data as well in the folder.


# Airplanes
config, _ = load_config("./configs/vae_airplane_latent.yaml")
dm = load_datamodule(config.data)
val_len = len(dm.test_ds)
torch.save(dm.train_ds[:val_len].ect, "./results/references/airplane_train_ect.pt")

# Car
config, _ = load_config("./configs/vae_car_latent.yaml")
dm = load_datamodule(config.data)
val_len = len(dm.test_ds)
torch.save(dm.train_ds[:val_len].ect, "./results/references/car_train_ect.pt")

# Chair
config, _ = load_config("./configs/vae_chair_latent.yaml")
dm = load_datamodule(config.data)
val_len = len(dm.test_ds)
print(val_len)
torch.save(dm.train_ds[:val_len].ect, "./results/references/chair_train_ect.pt")
