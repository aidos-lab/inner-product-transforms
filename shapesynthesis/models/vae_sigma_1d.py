from typing import Literal

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from layers.ect import EctConfig
from metrics.loss import compute_mse_kld_loss_beta_annealing_fn
from pyvista import filters
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.regression import MeanSquaredError
from torchvision.transforms import Compose

from shapesynthesis.datasets.transforms import EctTransform, RandomRotate
from shapesynthesis.layers import ect

torch.set_float32_matmul_precision("high")


class ModelConfig:
    """
    Base configuration for the VAE model and contains all parameters. The
    module provides the relative path to the file. The ECT config is the
    configuration of the ECT used for the input image. The latent dimension is
    the dimension of the VAE. The Beta min and Beta max are the are the minimum
    and maximum values of the weighting factor of the KL-divergence and the
    period (in epochs) is the period until the wight returns at its original
    value.
    """

    module: str
    latent_dim: int
    learning_rate: float
    ectconfig: EctConfig
    beta_period: int
    beta_min: float
    beta_max: float
    device: str = "cuda"
    model_type: str = "sigma_vae"
    img_channels: int = 3
    img_size: int = 128
    z_dim: int = 64
    filters_m: int = 32


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# class UnFlatten(nn.Module):
#     def __init__(self, n_channels):
#         super(UnFlatten, self).__init__()
#         # self.n_channels = n_channels
#
#     def forward(self, input):
#         # size = int((input.size(1) // self.n_channels))
#         # return input.view(input.size(0), self.n_channels, size)
#
#         # (B, 2 * W) -> (B, 8, W / 4)
#         # (B,4*C,W/4) -> (B,32,W/4)
#         return input.view(-1, 32, 32, 32)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        out = input.view(input.size(0), self.n_channels, size, size)
        return out


class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.device = "cuda"
        self.z_dim = 64
        self.img_channels = 1
        self.model = "sigma_vae"
        img_size = 128
        filters_m = 64

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, 1, 128, 128])
        print(self.encoder(demo_input).shape)
        h_dim = self.encoder(demo_input).shape[1]

        print("h_dim", h_dim)

        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc2 = nn.Linear(self.z_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)

        self.log_sigma = 0
        if self.model == "sigma_vae":
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(
                torch.full((1,), 0, dtype=torch.float32)[0],
                requires_grad=True,
            )

    @staticmethod
    def get_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(1, filters_m, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4 * filters_m, 4 * filters_m, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4 * filters_m, 4 * filters_m, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

    @staticmethod
    def get_decoder(filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(4 * filters_m),
            nn.ConvTranspose2d(4 * filters_m, 4 * filters_m, 6, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * filters_m, 4 * filters_m, 6, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, 6, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, 6, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, 1, 5, stride=1, padding=2),
            nn.Sigmoid(),
        )

    # @staticmethod
    # def get_encoder(img_channels, filters_m):
    #     return nn.Sequential(
    #         # Input is ECT with W=128 and C=128.
    #         # (B,C,W) -> (B,2*C,W)
    #         nn.Conv1d(img_channels, 2 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         # (B,2*C,W) -> (B,2*C,W)
    #         nn.Conv1d(2 * filters_m, 2 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         # (B, 2*C, W) -> (B,2*C,W / 2)
    #         nn.Conv1d(2 * filters_m, 2 * filters_m, 3, stride=2, padding=1),
    #         nn.SiLU(),
    #         # (B, 2*C, W/2) -> (B, 4*C,W / 2)
    #         nn.Conv1d(2 * filters_m, 4 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         # (B, 4*C, W/2) -> (B, 4*C,W / 2)
    #         nn.Conv1d(4 * filters_m, 4 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         # (B, 4*C, W/2) -> (B, 4*C,W / 4)
    #         nn.Conv1d(4 * filters_m, 4 * filters_m, 3, stride=2, padding=1),
    #         nn.SiLU(),
    #         # (B, 4*C, W / 4 ) -> (B, 8*C, W/4)
    #         nn.Conv1d(4 * filters_m, 8 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Conv1d(8 * filters_m, 8 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         # (B, 8*C, W/4) -> (B, 8, W / 4)
    #         nn.Conv1d(8 * filters_m, 32, 3, stride=1, padding=1),
    #         # (B, 8, W / 4) -> (B, 2 * W)
    #         Flatten(),
    #     )
    #
    # @staticmethod
    # def get_decoder(filters_m, out_channels):
    #     return nn.Sequential(
    #         # (B, 2 * W) -> (B, 8, W / 4)
    #         UnFlatten(32),
    #         nn.Conv1d(32, 8 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Conv1d(8 * filters_m, 8 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Conv1d(8 * filters_m, 4 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Conv1d(4 * filters_m, 4 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv1d(4 * filters_m, 2 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Conv1d(2 * filters_m, 2 * filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv1d(2 * filters_m, filters_m, 3, stride=1, padding=1),
    #         nn.SiLU(),
    #         nn.Conv1d(filters_m, filters_m, 3, stride=1, padding=1),
    #         nn.Sigmoid(),
    #     )

    def encode(self, x):
        # x = (x.movedim(-1, -2) + 1) / 2
        x = (x.unsqueeze(1) + 1) / 2
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # return 2 * self.decoder(self.fc2(z)).movedim(-1, -2) - 1
        out = 2 * self.decoder(self.fc2(z)).squeeze() - 1
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), None, mu, logvar

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1
        """

        if self.model == "gaussian_vae":
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == "sigma_vae":
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == "optimal_sigma_vae":
            log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = gaussian_nll(x_hat, log_sigma, x).sum()

        return rec

    def loss_function(self, recon_x, x, mu, logvar):

        # print("########################")
        # print(recon_x.shape)
        # print(x.shape)
        # print("########################")

        # x = (x.movedim(-1, -2) + 1) / 2
        # recon_x = (recon_x.movedim(-1, -2) + 1) / 2
        x = (x.unsqueeze(1) + 1) / 2
        recon_x = (recon_x.unsqueeze(1) + 1) / 2

        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == "mse_vae":
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return rec, kl


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )


###################################################
###  Lightning
###################################################


class BaseLightningModel(L.LightningModule):
    """Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.model = ConvVAE()
        # self.rotation_transform = Compose(
        #     [
        #         RandomRotate(axis=0),
        #         RandomRotate(axis=1),
        #         RandomRotate(axis=2),
        #     ]
        # )
        self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")
        # Metrics
        # self.train_fid = FrechetInceptionDistance(
        #     feature=64, normalize=True, input_img_size=(1, 128, 128)
        # )
        # self.val_fid = FrechetInceptionDistance(
        #     feature=64, normalize=True, input_img_size=(1, 128, 128)
        # )
        # self.sample_fid = FrechetInceptionDistance(
        #     feature=64, normalize=True, input_img_size=(1, 128, 128)
        # )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
        batch_len = len(pcs_gt)
        # pcs_gt = self.rotation_transform(pcs_gt)

        ect_gt = self.ect_transform(pcs_gt)

        recon_batch, _, mu, logvar = self(ect_gt)

        # Compute loss
        rec, kl = self.model.loss_function(recon_batch, ect_gt, mu, logvar)
        total_loss = rec + kl

        ###############################################################
        ### Metrics
        ###############################################################

        mse_loss = F.mse_loss(recon_batch, ect_gt)

        # if step == "train":
        #     # self.train_fid.update(
        #     #     (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
        #     # )
        #     # self.train_fid.update(
        #     #     (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
        #     # )
        #     self.train_fid.update(
        #         (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
        #     )
        #     self.train_fid.update(
        #         (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
        #     )
        #     fid = self.train_fid
        #     sample_fid = torch.tensor(0.0)
        # elif step == "validation":
        #     self.val_fid.update(
        #         (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
        #     )
        #     self.val_fid.update(
        #         (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
        #     )
        #
        #     samples = self.model.sample(n=batch_len)
        #     self.sample_fid.update(
        #         (samples.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
        #     )
        #     self.sample_fid.update(
        #         (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
        #     )
        #     fid = self.val_fid
        #     sample_fid = self.sample_fid
        # else:
        fid = torch.tensor(0.0)
        sample_fid = torch.tensor(0.0)

        loss_dict = {
            f"{step}_rec_loss": rec,
            f"{step}_kl_loss": kl,
            f"{step}_mse_loss": mse_loss,
            f"{step}_fid": fid,
            f"{step}_sample_fid": sample_fid,
            "loss": total_loss,
        }

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=len(pcs_gt),
            on_step=False,
            on_epoch=True,
        )

        return recon_batch, ect_gt, total_loss

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        _, _, loss = self.general_step(batch, batch_idx, "train")
        return loss
