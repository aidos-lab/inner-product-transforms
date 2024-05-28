from loggers import get_wandb_logger
import torch
import matplotlib.pyplot as plt

from models.base import BaseModel
from models.vae import VanillaVAE

from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_loss_fn


import lightning as L
from omegaconf import OmegaConf


from datasets.mnist import MnistDataModule
from datasets.config import MnistDataModuleConfig
from layers.ect import EctLayer, EctConfig
import torch


CUDA_DEVICE_NUM = 0
DEVICE = "cuda:0"

V = torch.vstack(
    [
        torch.sin(torch.linspace(0, 2 * torch.pi, 64, device=DEVICE)),
        torch.cos(torch.linspace(0, 2 * torch.pi, 64, device=DEVICE)),
    ]
)


layer = EctLayer(
    EctConfig(num_thetas=64, bump_steps=64, normalized=True, device=DEVICE), v=V
)


dm = MnistDataModule(MnistDataModuleConfig(root="./data/mnistpointcloud"))


model = VanillaVAE(in_channels=1, latent_dim=64)


litmodel = BaseModel(
    model,
    *get_mse_metrics(),
    accuracies_fn=compute_mse_accuracies,
    loss_fn=compute_mse_loss_fn,
    learning_rate=0.01,
    layer=layer,
)


metrics = get_mse_metrics()

litmodel2 = BaseModel.load_from_checkpoint(
    "./trained_models/vae.ckpt",
    model=model,
    training_accuracy=metrics[0],
    test_accuracy=metrics[1],
    validation_accuracy=metrics[2],
    accuracies_fn=compute_mse_accuracies,
    loss_fn=compute_mse_loss_fn,
    learning_rate=0.01,
    layer=layer,
)


data_loader = dm.train_dataloader()
device = DEVICE
unnormalizer = (None,)
figsize = (20, 2.5)
n_images = 10

fig, axes = plt.subplots(
    nrows=2, ncols=n_images, sharex=True, sharey=True, figsize=figsize
)

for batch_idx, features in enumerate(data_loader):
    break

ect = layer(features.to(device)).unsqueeze(1)

print(litmodel2.to(DEVICE))

with torch.no_grad():
    decoded, theinput, z_mean, z_log_var = litmodel2.forward(ect)


orig_images = ect[:n_images]

torch.save(decoded, "decoded.pt")
torch.save(ect, "ect.pt")
torch.save(features, "features.pt")


# for i in range(n_images):
#     for ax, img in zip(axes, [orig_images, decoded]):
#         curr_img = img[i].detach().to(torch.device("cpu"))
#         # print(curr_img)
#         ax[i].imshow(curr_img.view((64, 64)), cmap="binary")

# plt.show()


samples = litmodel2.model.sample(64, "cuda:0")
torch.save(samples, "samples.pt")
