from uu import decode
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_scatter import segment_coo
from sklearn.datasets import make_blobs
import itertools
from layers.ect import EctLayer
from layers.config import EctConfig
import time
from torch_geometric.data import Data, Batch
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.utils import barabasi_albert_graph
import torch.nn.functional as F
from datasets.mnist import MnistDataModule
from datasets.config import MnistDataModuleConfig

from notebooks.helper_plotting import plot_training_loss
from notebooks.helper_plotting import plot_generated_images
from model import VanillaVAE

NUM_PTS = 5
CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE_NUM}" if torch.cuda.is_available() else "cpu")

# Hyperparameters
RANDOM_SEED = 123
LEARNING_RATE = 1e-5
BATCH_SIZE = 128
NUM_EPOCHS = 100
LATENT_DIM = 128

V = torch.vstack(
    [
        torch.sin(torch.linspace(0, 2 * torch.pi, 64, device=DEVICE)),
        torch.cos(torch.linspace(0, 2 * torch.pi, 64, device=DEVICE)),
    ]
)

dataset = MnistDataModule(
    MnistDataModuleConfig(root="./data/mnistpointcloud", batch_size=BATCH_SIZE)
)

layer = EctLayer(
    EctConfig(num_thetas=64, bump_steps=64, normalized=True, device=DEVICE), v=V
)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :32, :32]


model = VanillaVAE(in_channels=1, latent_dim=3200)
model.to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features, reduction="sum")
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


train_loader = dataset.train_dataloader()
num_epochs = NUM_EPOCHS
model = model
optimizer = optimizer
device = DEVICE
skip_epoch_stats = True
logging_interval = 50
reconstruction_term_weight = 1

log_dict = {
    "train_combined_loss_per_batch": [],
    "train_combined_loss_per_epoch": [],
    "train_reconstruction_loss_per_batch": [],
    "train_kl_loss_per_batch": [],
}


loss_fn = F.mse_loss

start_time = time.time()
for epoch in range(num_epochs):

    model.train()
    for batch_idx, features in enumerate(train_loader):

        ect = layer(features.to(DEVICE)).unsqueeze(1)
        # FORWARD AND BACK PROP
        decoded, theinput, z_mean, z_log_var = model(ect)

        # total loss = reconstruction loss + KL divergence
        # kl_divergence = (0.5 * (z_mean**2 +
        #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
        kl_div = -0.5 * torch.sum(
            1 + z_log_var - z_mean**2 - torch.exp(z_log_var),
            axis=1,
        )  # sum over latent dimension

        batchsize = kl_div.size(0)
        kl_div = kl_div.mean()  # average over batch dimension

        pixelwise = loss_fn(decoded, ect, reduction="none")
        pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
        pixelwise = pixelwise.mean()  # average over batch dimension

        loss = reconstruction_term_weight * pixelwise + kl_div

        optimizer.zero_grad()

        loss.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        log_dict["train_combined_loss_per_batch"].append(loss.item())
        log_dict["train_reconstruction_loss_per_batch"].append(pixelwise.item())
        log_dict["train_kl_loss_per_batch"].append(kl_div.item())

        if not batch_idx % logging_interval:
            print(
                "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                % (epoch + 1, num_epochs, batch_idx, len(train_loader), loss)
            )

    if not skip_epoch_stats:
        model.eval()

        with torch.set_grad_enabled(False):  # save memory during inference

            train_loss = compute_epoch_loss_autoencoder(
                model, train_loader, loss_fn, device
            )
            print(
                "***Epoch: %03d/%03d | Loss: %.3f" % (epoch + 1, num_epochs, train_loss)
            )
            log_dict["train_combined_per_epoch"].append(train_loss.item())

    print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))

print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))


plot_training_loss(
    log_dict["train_reconstruction_loss_per_batch"],
    NUM_EPOCHS,
    custom_label=" (reconstruction)",
)
plot_training_loss(
    log_dict["train_kl_loss_per_batch"], NUM_EPOCHS, custom_label=" (KL)"
)
plot_training_loss(
    log_dict["train_combined_loss_per_batch"], NUM_EPOCHS, custom_label=" (combined)"
)
plt.show()


data_loader = dataset.train_dataloader()
device = DEVICE
unnormalizer = (None,)
figsize = (20, 2.5)
n_images = 10

fig, axes = plt.subplots(
    nrows=2, ncols=n_images, sharex=True, sharey=True, figsize=figsize
)

for batch_idx, features in enumerate(data_loader):

    features = layer(features.to(device)).unsqueeze(1)

    color_channels = features.shape[1]
    image_height = features.shape[2]
    image_width = features.shape[3]

    with torch.no_grad():
        decoded, theinput, z_mean, z_log_var = model(features)

    print(decoded.shape)

    orig_images = features[:n_images]
    decoded[:n_images]
    break

for i in range(n_images):
    for ax, img in zip(axes, [orig_images, decoded]):
        curr_img = img[i].detach().to(torch.device("cpu"))

        ax[i].imshow(curr_img.view((image_height, image_width)), cmap="binary")

plt.show()


torch.save(model.state_dict(), "model.pt")
