from typing import Literal

import lightning as L
import pydantic
import torch
from torch import nn

from layers.ect import EctConfig
from metrics.loss import compute_mse_kld_loss_fn


class ModelConfig(pydantic.BaseModel):
    module: str
    latent_dim: int
    lr: float
    ectconfig: EctConfig
    beta_period: int
    beta_min: float
    beta_max: float
    im_channels: int
    disc_start: int
    disc_weight: float
    perceptual_weight: float


class Model(nn.Module):
    """
    Standard implementation of a VAE.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        self.latent_dim = latent_dim

        filters_m = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, filters_m, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, 7, stride=2, padding=3),
            nn.Flatten(),
            # Output: (B, 4*F, W/8, H/8)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                4 * filters_m, 2 * filters_m, 7, stride=2, padding=3, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * filters_m, filters_m, 7, stride=2, padding=3, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, 1, 7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),
        )

        self.fc_mu = nn.Linear(4 * 32 * 16 * 16, latent_dim)
        self.fc_var = nn.Linear(4 * 32 * 16 * 16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 4 * 32 * 16 * 16)

        # self.encoder = nn.Sequential(
        #     # (B, 128, W) -> (B, 64, W/2)
        #     nn.Conv1d(128, 256, kernel_size=13, stride=2, padding=6),
        #     nn.LeakyReLU(),
        #     # (B, 64, W/2) -> (B, 32, W/4)
        #     nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=6),
        #     nn.LeakyReLU(),
        #     # (B, 64, W/2) -> (B, 32, W/4)
        #     nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=6),
        #     nn.LeakyReLU(),
        #     # (B, 64, W/2) -> (B, 32, W/4)
        #     nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=6),
        #     nn.LeakyReLU(),
        #     # (B, 32, W/4) -> (B, 16, W/8)
        #     nn.Conv1d(256, 64, kernel_size=13, stride=2, padding=6),
        #     # Output -> (B, 16, 16)
        # )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(
        #         64, 256, kernel_size=13, stride=1, padding=6, output_padding=0
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         256, 256, kernel_size=13, stride=1, padding=6, output_padding=0
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         256, 256, kernel_size=13, stride=1, padding=6, output_padding=0
        #     ),
        #     nn.Upsample(scale_factor=2),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         256, 256, kernel_size=13, stride=1, padding=6, output_padding=0
        #     ),
        #     nn.LeakyReLU(),
        #     nn.Upsample(scale_factor=2),
        #     nn.ConvTranspose1d(
        #         256, 128, kernel_size=13, stride=1, padding=6, output_padding=0
        #     ),
        #     nn.Tanh(),
        # )

    def encode(self, input_tensor):
        result = self.encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 4 * 32, 16, 16)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_tensor):
        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        decoded = self.decode(z)
        mse_loss = torch.nn.functional.mse_loss(input_tensor, decoded, reduction="mean")
        internal_loss = mse_loss + self.config.beta_min * kld_loss
        return decoded, internal_loss, {"mse_loss": mse_loss, "kld_los": kld_loss}

    def sample(self, n: int, device: str = "cuda"):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(n, self.latent_dim, device=device)

        samples = self.decode(z).movedim(-1, -2)
        return samples


# class BaseLightningModel(L.LightningModule):
#     """Base model for VAE models"""
#
#     def __init__(self, config: ModelConfig):
#         self.config = config
#         super().__init__()
#         self.training_accuracy = MeanSquaredError()
#         self.validation_accuracy = MeanSquaredError()
#         self.test_accuracy = MeanSquaredError()
#         # self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")
#         self.ect_transform = Ect2DTransform(config=config.ectconfig, device="cuda")
#
#         # # Metrics
#         # self.train_fid = FrechetInceptionDistance(
#         #     feature=64, normalize=True, input_img_size=(1, 128, 128)
#         # )
#         # self.val_fid = FrechetInceptionDistance(
#         #     feature=64, normalize=True, input_img_size=(1, 128, 128)
#         # )
#         # self.sample_fid = FrechetInceptionDistance(
#         #     feature=64, normalize=True, input_img_size=(1, 128, 128)
#         # )
#
#         self.model = VanillaVAE(config=self.config)
#
#         self.visualization = []
#
#         self.save_hyperparameters()
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=self.config.learning_rate
#         )
#         return optimizer
#
#     def forward(self, batch):  # pylint: disable=arguments-differ
#         x = self.model(batch)
#         return x
#
#     def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
#         pcs_gt = pcs_gt[0]
#         batch_len = len(pcs_gt)
#
#         ect_gt = self.ect_transform(pcs_gt)
#
#         recon_batch, _, mu, logvar = self(ect_gt)
#
#         total_loss, kl_loss, mse_loss = compute_mse_kld_loss_fn(
#             recon_batch, mu, logvar, ect_gt, beta=0.001
#         )
#
#         ###############################################################
#         ### Metrics
#         ###############################################################
#
#         # if step == "train":
#         #     self.train_fid.update(
#         #         (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
#         #     )
#         #     self.train_fid.update(
#         #         (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
#         #     )
#         #     fid = self.train_fid
#         #     sample_fid = torch.tensor(0.0)
#         # elif step == "validation":
#         #     self.val_fid.update(
#         #         (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
#         #     )
#         #     self.val_fid.update(
#         #         (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
#         #     )
#         #
#         #     samples = self.model.sample(n=batch_len, device="cuda")
#         #     self.sample_fid.update(
#         #         (samples.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
#         #     )
#         #     self.sample_fid.update(
#         #         (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
#         #     )
#         #     fid = self.val_fid
#         #     sample_fid = self.sample_fid
#         # else:
#         fid = torch.tensor(0.0)
#         sample_fid = torch.tensor(0.0)
#
#         loss_dict = {
#             f"{step}_kl_loss": kl_loss,
#             f"{step}_mse_loss": mse_loss,
#             f"{step}_fid": fid,
#             f"{step}_sample_fid": sample_fid,
#             "loss": total_loss,
#         }
#
#         self.log_dict(
#             loss_dict,
#             prog_bar=True,
#             batch_size=len(pcs_gt),
#             on_step=False,
#             on_epoch=True,
#         )
#
#         # return recon_batch, ect_gt, total_loss
#         return total_loss

# def general_step(
#     self, batch, batch_idx, step: Literal["train", "test", "validation"]
# ):
#
#
#     decoded, _, z_mean, z_log_var = self(ect)
#
#     # loss = compute_mse_loss_fn(decoded, ect)
#     loss_dict = compute_mse_kld_loss_beta_annealing_fn(
#         decoded,
#         z_mean,
#         z_log_var,
#         ect,
#         self.current_epoch,
#         period=self.config.beta_period,
#         beta_min=self.config.beta_min,
#         beta_max=self.config.beta_max,
#         prefix=step + "_",
#     )
#
#     self.log_dict(
#         loss_dict,
#         prog_bar=True,
#         batch_size=len(batch),
#         on_step=False,
#         on_epoch=True,
#     )
#
#     return loss_dict[f"{step}_loss"]
#
# def test_step(self, batch, batch_idx):
#     return self.general_step(batch, batch_idx, "test")
#
# def validation_step(self, batch, batch_idx):
#     return self.general_step(batch, batch_idx, "validation")
#
# def training_step(self, batch, batch_idx):
#     return self.general_step(batch, batch_idx, "train")
