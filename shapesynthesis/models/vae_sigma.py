"""
Implementation of the SigmaVAE.
Is based on the models that already trained and
evaluated.
"""

# For evaluating / debugging the model inline (without the
# dataset, etc).
if __name__ == "__main__":
    import sys

    sys.path.append("./shapesynthesis")

from typing import Literal

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from layers.ect import EctConfig
from metrics.loss import compute_mse_kld_loss_beta_annealing_fn
from pydantic import BaseModel
from torch import nn
from torchmetrics.regression import MeanSquaredError


class ModelConfig(BaseModel):
    """
    Base configuration for the VAE model and contains all parameters. The
    module provides the relative path to the file. The ECT config is the
    configuration of the ECT used for the input image. The latent dimension is
    the dimension of the VAE. The Beta min and Beta max are the are the minimum
    and maximum values of the weighting factor of the KL-divergence and the
    period (in epochs) is the period until the wight returns at its original
    value.
    """

    latent_dim: int
    learning_rate: float
    ectconfig: EctConfig
    beta_period: int
    beta_min: float
    beta_max: float
    device: str = "cuda"
    model_type: str = "sigma_vae"
    img_channels: int = 3
    img_size: int = 256
    z_dim: int = 64
    filters_m: int = 32


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels))
        return input.view(input.size(0), self.n_channels, size)


def loss_function(recon_x, x, mu, logvar):
    # Important: both reconstruction and KL divergence loss have to be summed over all element!
    # Here we also sum the over batch and divide by the number of elements in the data later
    rec = self.reconstruction_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return rec, kl


class SigmaVAE(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.z_dim = config.z_dim
        self.img_channels = config.img_channels
        self.model_type = config.model_type
        filters_m = config.filters_m

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, img_size])
        h_dim = self.encoder(demo_input).shape[1]
        print("h_dim", h_dim)

        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc2 = nn.Linear(self.z_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)

        self.log_sigma = 0
        if self.model_type == "sigma_vae":
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(
                torch.full((1,), 0, dtype=torch.float32)[0],
                requires_grad=True,
            )

    @staticmethod
    def get_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv1d(img_channels, filters_m, 15, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(filters_m, 2 * filters_m, 45, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(2 * filters_m, 2 * filters_m, 45, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(2 * filters_m, 2 * filters_m, 45, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(2 * filters_m, 4 * filters_m, 55, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(4 * filters_m, 8 * filters_m, 65, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
        )

    @staticmethod
    def get_decoder(filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(8 * filters_m),
            nn.ConvTranspose1d(8 * filters_m, 4 * filters_m, 65, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(4 * filters_m, 2 * filters_m, 55, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(2 * filters_m, 2 * filters_m, 45, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(2 * filters_m, 2 * filters_m, 45, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(2 * filters_m, filters_m, 45, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(filters_m, out_channels, 15, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.fc2(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n):
        sample = torch.randn(n, self.z_dim, device=self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1
        """

        if self.model_type == "gaussian_vae":
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model_type == "sigma_vae":
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model_type == "optimal_sigma_vae":
            log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = gaussian_nll(x_hat, log_sigma, x).sum()

        return rec


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )


class BaseLightningModel(L.LightningModule):
    """Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()

        self.model = SigmaVAE(config=self.config)

        self.visualization = []

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):

        ect = batch.ect * 2 - 1

        decoded, _, z_mean, z_log_var = self(ect)

        # loss = compute_mse_loss_fn(decoded, ect)
        loss_dict = compute_mse_kld_loss_beta_annealing_fn(
            decoded,
            z_mean,
            z_log_var,
            ect,
            self.current_epoch,
            period=self.config.beta_period,
            beta_min=self.config.beta_min,
            beta_max=self.config.beta_max,
            prefix=step + "_",
        )

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )

        return loss_dict[f"{step}_loss"]

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "validation")
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")


if __name__ == "__main__":
    config = ModelConfig(
        latent_dim=64,
        learning_rate=0.1,
        ectconfig=EctConfig(
            num_thetas=256,
            resolution=256,
            r=1.1,
            scale=2,
            ect_type="points",
            ambient_dimension=3,
            normalized=True,
            seed=2024,
        ),
        beta_period=1,
        beta_min=1,
        beta_max=1,
    )
    model = VanillaVAE(config)
    ect = torch.rand(size=(2, 256, 256))
    decoded, _, _, _ = model(ect)
    print("ECT SHAPE", ect.shape)
    print("Decoded", decoded.shape)
    assert decoded.shape == ect.shape
