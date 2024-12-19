from typing import List, Literal, TypeAlias

import torch
from torch import nn
import lightning as L
from torchmetrics.regression import MeanSquaredError
import matplotlib.pyplot as plt
from pydantic import BaseModel


from layers.ect import EctConfig, EctLayer
from layers.directions import generate_directions
from metrics.loss import compute_mse_kld_loss_fn, compute_mse_loss_fn


class ModelConfig(BaseModel):
    in_channels: int
    latent_dim: int
    learning_rate: float
    ectconfig: EctConfig


class VanillaVAE(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        in_channels = config.in_channels
        latent_dim = config.latent_dim
        hidden_dims = [32, 64, 128, 256, 512]

        self.latent_dim = latent_dim

        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        with torch.no_grad():
            out = self.encoder(
                torch.zeros(
                    3,
                    1,
                    config.ectconfig.num_thetas,
                    config.ectconfig.resolution,
                )
            )
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
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
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
        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input_tensor, mu, log_var]

    def sample(self, num_samples: int, device: str):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class BaseLightningModel(L.LightningModule):
    """ "Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()

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

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):
        # ect = self.layer(batch, batch.batch).unsqueeze(1) * 2 - 1

        ect = batch.ect.unsqueeze(1) * 2 - 1

        decoded, _, z_mean, z_log_var = self(ect)

        # loss = compute_mse_loss_fn(decoded, ect)
        loss, kld_loss, mse_loss = compute_mse_kld_loss_fn(
            decoded, z_mean, z_log_var, ect, beta=0.000001
        )

        self.log_dict(
            {
                f"{step}_loss": loss.item(),
                f"{step}_kld_loss": kld_loss.item(),
                f"{step}_mse_loss": mse_loss.item(),
                # f"{step}_beta": beta,
            },
            prog_bar=True,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        # if batch_idx == 0 and step == "validation":
        #     self.visualization = [(ect, decoded)]

        return loss

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "validation")
        # if self.current_epoch % 10 == 0 and self.visualization:
        #     tensorboard_logger = self.logger.experiment
        #     ect, decoded = self.visualization[0]
        #     fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(16, 4))
        #     fig.tight_layout()
        #     for axis, orig, pred in zip(axes.T, ect.squeeze(), decoded.squeeze()):
        #         ax = axis[0]
        #         ax.imshow(orig.detach().cpu().numpy())
        #         ax.axis("off")
        #         ax = axis[1]
        #         ax.imshow(pred.detach().cpu().numpy())
        #         ax.axis("off")

        #     # Adding plot to tensorboard
        #     tensorboard_logger.add_figure(
        #         "reconstruction", plt.gcf(), global_step=self.global_step
        #     )

        #     samples = self.model.sample(8, "cuda:0")
        #     fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(16, 2))
        #     fig.tight_layout()
        #     for ax, s in zip(axes, samples.squeeze()):
        #         ax.imshow(s.detach().cpu().numpy())
        #         ax.axis("off")

        #     # Adding plot to tensorboard
        #     tensorboard_logger.add_figure(
        #         "samples", plt.gcf(), global_step=self.global_step
        #     )

        # self.visualization.clear()
        return loss

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")
