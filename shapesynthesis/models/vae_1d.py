if __name__ == "__main__":
    import sys

    sys.path.append("./shapesynthesis")

from dataclasses import dataclass
from typing import Literal

import lightning as L
import torch
from layers.ect import EctConfig
from metrics.loss import compute_mse_kld_loss_fn
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.regression import MeanSquaredError
from torchvision.transforms import Compose

from shapesynthesis.datasets.transforms import EctTransform, RandomRotate
from shapesynthesis.layers import ect


@dataclass
class ModelConfig:
    module: str
    latent_dim: int
    learning_rate: float
    ectconfig: EctConfig
    beta_period: int
    beta_min: float
    beta_max: float


class VanillaVAE(nn.Module):
    """
    Standard implementation of a VAE.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        in_channels = config.ectconfig.num_thetas
        latent_dim = config.latent_dim
        # hidden_dims = config.hidden_dims
        hidden_dims = [128, 256, 512, 1024]
        self.latent_dim = latent_dim

        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=11,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        with torch.no_grad():
            input_tensor = torch.zeros(
                3,
                config.ectconfig.num_thetas,
                config.ectconfig.resolution,
            )
            out = self.encoder(input_tensor)
            print(input_tensor.shape)
            print(out.shape)
        self.conv_out_shape = torch.tensor(out.shape[1:])
        print(self.conv_out_shape)
        self.conv_out_size = int(torch.prod(self.conv_out_shape))

        self.fc_mu = nn.Linear(self.conv_out_size, latent_dim)
        self.fc_var = nn.Linear(self.conv_out_size, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.conv_out_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=11,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=11,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_dims[-1],
                out_channels=self.config.ectconfig.num_thetas,
                kernel_size=11,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, input_tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
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
        result = result.view(-1, *self.conv_out_shape)
        result = self.decoder(result)

        result = self.final_layer(result)

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
        mu, log_var = self.encode(input_tensor.movedim(-1, -2).squeeze())
        z = self.reparameterize(mu, log_var)
        return [self.decode(z).movedim(-1, -2), input_tensor, mu, log_var]

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


class BaseLightningModel(L.LightningModule):
    """Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()
        self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")
        # Metrics
        self.train_fid = FrechetInceptionDistance(
            feature=64, normalize=True, input_img_size=(1, 128, 128)
        )
        self.val_fid = FrechetInceptionDistance(
            feature=64, normalize=True, input_img_size=(1, 128, 128)
        )
        self.sample_fid = FrechetInceptionDistance(
            feature=64, normalize=True, input_img_size=(1, 128, 128)
        )

        self.model = VanillaVAE(config=self.config)

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

    def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
        batch_len = len(pcs_gt)
        ect_gt = self.ect_transform(pcs_gt)

        recon_batch, _, mu, logvar = self(ect_gt)

        total_loss, kl_loss, mse_loss = compute_mse_kld_loss_fn(
            recon_batch, mu, logvar, ect_gt, beta=0.0001
        )

        ###############################################################
        ### Metrics
        ###############################################################

        if step == "train":
            self.train_fid.update(
                (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
            )
            self.train_fid.update(
                (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
            )
            fid = self.train_fid
            sample_fid = torch.tensor(0.0)
        elif step == "validation":
            self.val_fid.update(
                (recon_batch.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
            )
            self.val_fid.update(
                (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
            )

            samples = self.model.sample(n=batch_len, device="cuda")
            self.sample_fid.update(
                (samples.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=False
            )
            self.sample_fid.update(
                (ect_gt.unsqueeze(1).repeat(1, 3, 1, 1) + 1) / 2, real=True
            )
            fid = self.val_fid
            sample_fid = self.sample_fid
        else:
            fid = torch.tensor(0.0)
            sample_fid = torch.tensor(0.0)

        loss_dict = {
            f"{step}_kl_loss": kl_loss,
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

        # return recon_batch, ect_gt, total_loss
        return total_loss

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

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
