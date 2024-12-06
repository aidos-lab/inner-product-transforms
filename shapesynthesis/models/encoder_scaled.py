"""
For the dynamic loading of the models, the model class has to be called 
BaseModel. 
"""

from typing import TypeAlias, Literal

import torch
from torch import nn
from torchmetrics.regression import MeanSquaredError


from layers.directions import generate_uniform_directions
from layers.ect import EctLayer
import torch.nn.functional as F

from kaolin.metrics.pointcloud import chamfer_distance
import lightning as L


Tensor: TypeAlias = torch.Tensor


class BaseModel(L.LightningModule):
    def __init__(self, ectconfig, ectlossconfig, modelconfig):
        super().__init__()

        self.ectconfig = ectconfig
        self.ectlossconfig = ectlossconfig
        self.modelconfig = modelconfig

        self.loss_layer = EctLayer(
            ectlossconfig,
            v=generate_uniform_directions(
                num_thetas=64, seed=ectlossconfig.seed
            ).cuda(),
        )

        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()
        self.loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(modelconfig.ect_size**2, modelconfig.hidden_size),
            nn.ReLU(),
            nn.Linear(modelconfig.hidden_size, modelconfig.hidden_size),
            nn.ReLU(),
            nn.Linear(modelconfig.hidden_size, modelconfig.hidden_size),
            nn.ReLU(),
            nn.Linear(modelconfig.hidden_size, modelconfig.hidden_size),
            nn.ReLU(),
            nn.Linear(
                modelconfig.hidden_size,
                modelconfig.num_dims * modelconfig.num_pts,
            ),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(
        self, batch, _, step: Literal["train", "test", "validation"]
    ):
        batch_len = len(batch)
        _batch = batch.clone()
        __batch = batch.clone()
        __batch.x = __batch.x + 0.02 * torch.randn_like(__batch.x)

        ect = self.layer(__batch, __batch.batch)

        # # PLAYING WITH DIMENSION
        # _batch.x = self(ect.movedim(-1, -2)).view(-1, self.num_dims)

        # PLAYING WITH DIMENSION
        _batch.x = self(ect).view(-1, self.num_dims)

        _batch.batch = torch.arange(
            batch.batch.max().item() + 1, device=self.device
        ).repeat_interleave(self.num_pts)

        loss_ch = chamfer_distance(
            batch.x.view(batch_len, -1, self.num_dims),
            _batch.x.view(-1, self.num_pts, self.num_dims),
        ).mean()

        ect_pred = self.loss_layer(_batch, _batch.batch)
        ect_target = self.loss_layer(batch, batch.batch)

        eps = 10e-5
        ect_pred /= ect_pred.sum(axis=1, keepdim=True)
        ect_target /= ect_target.sum(axis=1, keepdim=True)
        ect_pred += eps
        ect_target += eps

        ect_kld_loss = (
            F.kl_div(ect_pred.log(), ect_target, None, None, reduction="none")
            .sum(dim=-1)
            .sum(dim=-1)
            / 2048
        )

        loss = ect_kld_loss.mean() + loss_ch
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        # self.log_accuracies(ect_hat, ect, batch_len, step)
        return loss

    def on_train_epoch_end(self) -> None:
        self.loss_layer.v = generate_uniform_directions(
            d=3, num_thetas=64
        ).cuda()
        return super().on_train_epoch_end()

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")
