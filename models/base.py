from typing import Literal
import lightning as L
import torch
from torchmetrics.regression import MeanSquaredError


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
        self.mse = MeanSquaredError()

    def forward(self, batch):
        x = self.model(batch)
        return x

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):
        batch_len = len(batch.y)
        ect = self.layer(batch).unsqueeze(1)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

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
