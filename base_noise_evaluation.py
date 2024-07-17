# %%
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
import numpy as np


from models.vae import VanillaVAE
from models.vae import BaseModel as BaseVAE

from models.encoder import BaseModel as EctEncoder

from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_loss_fn

from datasets.mnist import DataModule, DataModuleConfig
from layers.ect import EctLayer, EctConfig
from layers.directions import generate_directions

from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import adjust_sharpness

DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 10, "c": ".5"}

encoder_config = OmegaConf.load("./configs/config_encoder_mnist.yaml")
vae_config = OmegaConf.load("./configs/config_vae_mnist.yaml")

# %%
layer = EctLayer(
    EctConfig(
        num_thetas=encoder_config.layer.ect_size,
        bump_steps=encoder_config.layer.ect_size,
        normalized=True,
        device=DEVICE,
    ),
    v=generate_directions(
        encoder_config.layer.ect_size, encoder_config.layer.dim, DEVICE
    ),
)

dm = DataModule(DataModuleConfig(root="./data/mnistpointcloud"))


vae_model = VanillaVAE(
    in_channels=vae_config.model.in_channels,
    latent_dim=vae_config.model.latent_dim,
    img_size=vae_config.layer.ect_size,
)

# Load the VAE model
metrics = get_mse_metrics()

vae_litmodel = BaseVAE.load_from_checkpoint(
    f"./trained_models/{vae_config.model.save_name}",
    model=vae_model,
    training_accuracy=metrics[0],
    test_accuracy=metrics[1],
    validation_accuracy=metrics[2],
    accuracies_fn=compute_mse_accuracies,
    loss_fn=compute_mse_loss_fn,
    learning_rate=0.01,
    layer=layer,
).to(DEVICE)


# Load the encoder

ect_encoder_litmodel = EctEncoder.load_from_checkpoint(
    f"./trained_models/{encoder_config.model.save_name}",
    layer=layer,
    ect_size=encoder_config.layer.ect_size,
    hidden_size=encoder_config.model.hidden_size,
    num_pts=encoder_config.model.num_pts,
    num_dims=encoder_config.model.num_dims,
    learning_rate=encoder_config.model.learning_rate,
).to(DEVICE)

# %%

y = dm.test_ds[:].y

print(torch.unique(y))

idxs = torch.hstack(
    [
        # torch.where(y==1)[0][:1],
        # torch.where(y==2)[0][:1],
        torch.where(y == 2)[0][:1],
    ]
)


data_list = []
for idx in idxs:
    data_list.append(dm.test_ds[idx])


batch = Batch.from_data_list(data_list).to(DEVICE)

# %%
batch.to(DEVICE)
ect = layer(batch, batch.batch)

print(ect.shape)

blur = GaussianBlur(kernel_size=5, sigma=2)
ect_batch = []
ect_intermediary = ect
for idx in range(10):
    ect_intermediary = blur(ect_intermediary)
    ect_batch.append(ect_intermediary)


ect_batch = torch.stack(ect_batch)
ect_batch = ect_batch.unsqueeze(1)
print(ect_batch.shape)

# %%
# fig, axes = plt.subplots(2,10,figsize=(20,2))

# for blurred_img,ax in zip(ect_batch,axes.T):
#     ax[1].imshow(blurred_img.cpu().squeeze().detach().numpy())
#     ax[1].axis("off")

# %%


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


with torch.no_grad():
    recon_batch = ect_encoder_litmodel.model.forward(ect_batch)

recon_batch.shape

# ect_batch.shape

# %%
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(16, 2))
fig.subplots_adjust(wspace=0.05, hspace=0.05)

for idx, ax in enumerate(axes.T):

    # print(recon_pts_x.shape)
    recon_pts = recon_batch[idx].view(-1, 2).cpu().detach().squeeze().numpy()

    recon_pts = rotate(recon_pts, degrees=-90)

    ax[0].scatter(recon_pts[:, 0], recon_pts[:, 1], **PC_PLOT_CONFIG)
    ax[0].set_xlim([-1, 1])
    ax[0].set_ylim([-1, 1])
    ax[0].set_aspect(1)
    ax[0].axis("off")

    ax[1].imshow(ect_batch[idx].cpu().detach().squeeze().numpy(), **ECT_PLOT_CONFIG)
    ax[1].axis("off")

# plt.savefig("./figures/img/dsprites/reconstructed_pointcloud.svg",transparent=True)

# %%
