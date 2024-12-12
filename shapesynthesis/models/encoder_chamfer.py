from typing import TypeAlias, Literal
import torch
from torch import nn
import lightning as L

from metrics.loss import chamfer2D, chamfer3D


Tensor: TypeAlias = torch.Tensor


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(512, 256, kernel_size=3, stride=1),
            # nn.BatchNorm1d(num_features=256),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            # nn.Conv1d(256, 128, kernel_size=3, stride=1),
        )

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7168, 3 * 2048),
            nn.ReLU(),
            nn.Linear(3 * 2048, 3 * 2048),
            nn.Tanh(),
            nn.Linear(3 * 2048, 2048 * 3),
        )

    def forward(self, ect):
        ect = 2 * ect - 1
        ect = ect.movedim(-1, -2)
        x = self.conv(ect)
        x = self.layer(x.flatten(start_dim=1))
        return x


class BaseLightningModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = Model()
        self.count = 0

        if config.num_dims == 3:
            self.loss_fn = chamfer3D
        elif config.num_dims == 2:
            self.loss_fn = chamfer2D
        else:
            raise ValueError(f"Number of dimensions {config.num_dims} not supported")

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def forward(self, ect):  # pylint: disable=arguments-differ
        x = self.model(ect)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
        batch_len = len(batch)
        pc_shape = batch[0].x.shape

        x = self(batch.ect).reshape(-1, self.config.num_dims)

        loss = self.loss_fn(
            x.view(-1, pc_shape[0], pc_shape[1]),
            batch.x.view(-1, pc_shape[0], pc_shape[1]),
        )
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")
