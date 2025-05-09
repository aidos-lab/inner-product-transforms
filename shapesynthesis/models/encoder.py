import functools
import operator
from typing import Literal, TypeAlias

import lightning as L
import torch
from layers.directions import generate_uniform_directions
from layers.ect import EctConfig, EctLayer
from metrics.loss import chamfer2DECT, chamfer3DECT
from pydantic import BaseModel
from torch import nn

Tensor: TypeAlias = torch.Tensor


class ModelConfig(BaseModel):
    """
    Base configuration for the model and contains all parameters. The module
    provides the relative path to the file. Number of points determines the
    number of points that the model outputs. The ECT config is the configuration
    of the ECT used for the input image and the ECT Loss configuration is the
    config for the ECT used to calculate the loss between the reconstructed
    point cloud and the ground truth.
    """

    module: str
    num_pts: int
    learning_rate: float
    ectlossconfig: EctConfig
    ectconfig: EctConfig


class Model(nn.Module):
    """
    The core model that reconstructs an ECT back into a point cloud.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
        )

        # Function to calculate the shape of the CNN output, in order to
        # initialize the linear layers without having to adjust the input
        # dimension manually.
        num_cnn_features = functools.reduce(
            operator.mul,
            list(
                self.conv(
                    torch.rand(
                        1, config.ectconfig.num_thetas, config.ectconfig.resolution
                    )
                ).shape
            ),
        )

        # Ambient dimension is 3 for 3D and 2 for 2D point clouds.
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                num_cnn_features, config.ectconfig.ambient_dimension * config.num_pts
            ),
            nn.ReLU(),
            nn.Linear(
                config.ectconfig.ambient_dimension * config.num_pts,
                config.ectconfig.ambient_dimension * config.num_pts,
            ),
            nn.Tanh(),
            nn.Linear(
                config.ectconfig.ambient_dimension * config.num_pts,
                config.ectconfig.ambient_dimension * config.num_pts,
            ),
        )

    def forward(self, ect):
        """
        We compute the forward pass here. The input ECT is viewed as a image and
        each pixel has values between [0,1]. We rescale to [-1,1] to accommodat
        the CNN layers, who prefer this type of input. Lastly, the Tanh
        activation function at the end ensures that the models output is
        relatively bounded.
        """
        ect = 2 * ect - 1
        ect = ect.movedim(-1, -2)
        x = self.conv(ect)
        x = self.layer(x.flatten(start_dim=1))
        return x


class BaseLightningModel(L.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.model = Model(config)
        self.losslayer = EctLayer(
            config.ectlossconfig,
            v=generate_uniform_directions(
                config.ectlossconfig.num_thetas,
                d=config.ectlossconfig.ambient_dimension,
                seed=config.ectlossconfig.seed,
            ).cuda(),
        )

        # Determine the loss function based on dimension.
        # The 2D chamfer first embeds the 2d into 3d.
        if config.ectlossconfig.ambient_dimension == 3:
            self.loss_fn = chamfer3DECT
        elif config.ectlossconfig.ambient_dimension == 2:
            self.loss_fn = chamfer2DECT
        else:
            raise ValueError(
                f"Number of dimensions {config.ectlossconfig.ambient_dimension} not supported"
            )

        self.save_hyperparameters()

        self.plot_batch = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def forward(self, ect):
        x = self.model(ect)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
        """
        We clone the batch, the predict the reconstruction and compute the ECT
        of both the predicted and the ground truth point clouds. While the
        ground truth ect could be computed beforehand, for small
        resolution/directions (64x64) in our case, it does not slow down too
        much. For larger (128x128 or 256x256) it would be recommended to do
        this.
        """
        batch_len = len(batch)
        pc_shape = batch[0].x.shape
        _batch = batch.clone()

        _batch.x = self(batch.ect).reshape(
            -1, self.config.ectlossconfig.ambient_dimension
        )

        _batch.batch = torch.arange(
            batch.batch.max().item() + 1, device=self.device
        ).repeat_interleave(self.config.num_pts)

        ect_pred = self.losslayer(_batch, _batch.batch)
        ect_gt = self.losslayer(batch, batch.batch)

        loss, ect_loss, cd_loss = self.loss_fn(
            _batch.x.view(
                -1, self.config.num_pts, self.config.ectconfig.ambient_dimension
            ),
            batch.x.view(-1, pc_shape[0], pc_shape[1]),
            ect_gt,
            ect_pred,
        )
        self.log_dict(
            {
                f"{step}_loss": loss,
                f"{step}_ect_loss": ect_loss,
                f"{step}_cd_loss": cd_loss,
            },
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")
