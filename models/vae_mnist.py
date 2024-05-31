import torch
from torch import nn
import lightning as L

from typing import List, Any, Literal, TypeAlias

from torchmetrics.regression import MeanSquaredError

Tensor: TypeAlias = torch.Tensor


class BaseModel(L.LightningModule):
    def __init__(
        self,
        model,
        training_accuracy,
        test_accuracy,
        validation_accuracy,
        accuracies_fn,
        loss_fn,
        learning_rate,
        layer,
    ):
        super().__init__()
        self.training_accuracy = training_accuracy
        self.validation_accuracy = validation_accuracy
        self.test_accuracy = test_accuracy
        self.loss_fn = loss_fn
        self.accuracies_fn = accuracies_fn
        self.model = model
        self.learning_rate = learning_rate
        self.layer = layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
        x = self.model(batch)
        return x

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):
        batch_len = len(batch.y)
        ect = self.layer(batch, batch.batch).unsqueeze(1) * 2 - 1
        decoded, _, z_mean, z_log_var = self(ect)
        # Squeeze x_hat to match the shape of y

        loss = self.loss_fn(decoded, z_mean, z_log_var, ect)
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        self.log_accuracies(decoded, ect, batch_len, step)
        return loss

    def log_accuracies(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
        if step == "train":
            accuracies = self.accuracies_fn(
                self.training_accuracy,
                x_hat.reshape(batch_len, -1),
                y.reshape(batch_len, -1),
                "train_accuracy",
            )
            for accuracy in accuracies:
                self.log(
                    **accuracy,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )
        elif step == "test":
            accuracies = self.accuracies_fn(
                self.test_accuracy,
                x_hat.reshape(batch_len, -1),
                y.reshape(batch_len, -1),
                "test_accuracy",
            )
            for accuracy in accuracies:
                self.log(
                    **accuracy,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )
        elif step == "validation":
            accuracies = self.accuracies_fn(
                self.validation_accuracy,
                x_hat.reshape(batch_len, -1),
                y.reshape(batch_len, -1),
                "validation_accuracy",
            )
            for accuracy in accuracies:
                self.log(
                    **accuracy,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_len,
                )

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")


class VanillaVAE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        latent_dim: int,
        hidden_dims: List | None = None,
        **kwargs,
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

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
            out = self.encoder(torch.zeros(2, 1, img_size, img_size))
        self.conv_out_shape = torch.tensor(out.shape[1:])
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

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def sample(self, num_samples: int, device: str, **kwargs) -> Tensor:
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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
