import torch
from torch import nn
import lightning as L
from typing import List, Any, Literal, TypeAlias
from torch_geometric.data import Batch, Data

from typing import TypeAlias


# from torch import tensor as Tensor

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
        _batch = batch.clone()

        ect = self.layer(batch, batch.batch)

        _batch.x = self(ect).view(-1, 2)

        _batch.batch = torch.arange(
            batch.batch.max().item() + 1, device=self.device
        ).repeat_interleave(100)

        ect_hat = self.layer(_batch, _batch.batch)
        loss = self.loss_fn(ect_hat, ect)
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


class EctEncoder(nn.Module):
    def __init__(self, num_pts: int, ect_size: int, hidden_size: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ect_size**2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * num_pts),
        )

    def forward(self, ect: Tensor) -> Tensor:
        return self.encoder(ect)
