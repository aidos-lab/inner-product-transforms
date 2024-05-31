import torch
from torch import nn
import lightning as L
from typing import Literal, TypeAlias

from typing import TypeAlias

from torchmetrics.regression import MeanSquaredError

from layers.ect import EctConfig, EctLayer

from kaolin.metrics.pointcloud import chamfer_distance

Tensor: TypeAlias = torch.Tensor


class BaseModel(L.LightningModule):
    def __init__(self, layer, ect_size, hidden_size, num_pts, learning_rate, num_dims):
        super().__init__()

        self.learning_rate = learning_rate
        self.layer = layer
        self.num_dims = num_dims
        self.num_pts = num_pts

        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()
        self.loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ect_size**2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_dims * num_pts),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
        x = self.model(batch)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
        batch_len = len(batch.y)
        _batch = batch.clone()

        ect = self.layer(batch, batch.batch)

        _batch.x = self(ect).view(-1, self.num_dims)

        _batch.batch = torch.arange(
            batch.batch.max().item() + 1, device=self.device
        ).repeat_interleave(self.num_pts)

        ect_hat = self.layer(_batch, _batch.batch)

        loss_ch = chamfer_distance(
            batch.x.view(batch_len, -1, self.num_dims),
            _batch.x.view(-1, self.num_pts, self.num_dims),
        ).mean()

        loss = loss_ch + self.loss_fn(ect_hat, ect)
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        self.log_accuracies(ect_hat, ect, batch_len, step)
        return loss

    def log_accuracies(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
        if step == "train":
            accuracy_fn = self.training_accuracy
            name = "train_accuracy"
        elif step == "test":
            accuracy_fn = self.test_accuracy
            name = "test_accuracy"
        elif step == "validation":
            accuracy_fn = self.validation_accuracy
            name = "validation_accuracy"

        accuracy = accuracy_fn(
            x_hat.reshape(batch_len, -1),
            y.reshape(batch_len, -1),
        )
        self.log(
            name,
            accuracy,
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
