import functools
import operator
from dataclasses import dataclass
from typing import Literal, TypeAlias

import lightning as L
import torch
from layers.directions import generate_uniform_directions
from layers.ect import EctConfig, EctLayer
from metrics.loss import chamfer2DECT, chamfer3DECT
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose

from shapesynthesis.datasets.transforms import EctTransform, RandomRotate

Tensor: TypeAlias = torch.Tensor


@dataclass
class ModelConfig:
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
            ###########################################################
            nn.Conv1d(
                config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                4 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=4 * config.ectconfig.num_thetas),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                4 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=8 * config.ectconfig.num_thetas),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                8 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
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
        ect = ect.movedim(-1, -2)
        x = self.conv(ect)
        x = self.layer(x.flatten(start_dim=1))
        return x


class BaseLightningModel(L.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pred_test_batches = []
        self.gt_test_batches = []

        self.model = Model(config)
        self.losslayer = EctTransform(config=config.ectlossconfig, device="cuda")
        self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")

        print("CONIFG")
        print(self.losslayer.config)

        self.rotation_transform = Compose(
            [
                RandomRotate(axis=0),
                RandomRotate(axis=1),
                RandomRotate(axis=2),
            ]
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        # Define the learning rate scheduler
        # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

        # return [optimizer], [
        #     {"scheduler": scheduler, "interval": "step", "monitor": "lr-Adam"}
        # ]
        return [optimizer], [scheduler]
        # return optimizer

    def forward(self, batch):
        x = self.model(batch)
        return x

    def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
        batch_len = len(pcs_gt)
        # pcs_gt = self.rotation_transform(pcs_gt)

        ect_gt = self.ect_transform(pcs_gt)
        pcs_pred = self(ect_gt).reshape(
            batch_len, self.config.num_pts, self.config.ectconfig.ambient_dimension
        )

        ect_loss_pred = self.losslayer(pcs_pred)
        ect_loss_gt = self.losslayer(pcs_gt)

        loss, ect_loss, cd_loss = self.loss_fn(
            pcs_pred,
            pcs_gt,
            ect_loss_pred,
            ect_loss_gt,
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
        return pcs_gt, pcs_pred, loss

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):
        _, _, loss = self.general_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self.general_step(batch, batch_idx, "validation")
        return loss
