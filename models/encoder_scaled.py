from typing import TypeAlias, Literal

import torch
from torch import nn
import lightning as L

from kaolin.metrics.pointcloud import chamfer_distance

from torchmetrics.regression import MeanSquaredError
from layers.ect import EctConfig, EctLayer

import torch.nn.functional as F

Tensor: TypeAlias = torch.Tensor


def generate_uniform_directions(num_thetas: int = 64, d: int = 3):
    """
    Generate randomly sampled directions from a sphere in d dimensions.

    First a standard gaussian centered at 0 with standard deviation 1 is sampled
    and then projected onto the unit sphere. This yields a uniformly sampled set
    of points on the unit spere. Please note that the generated shapes with have
    shape [d, num_thetas].

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    v = torch.randn(size=(d, num_thetas))
    v /= v.pow(2).sum(axis=0).sqrt().unsqueeze(1).T
    return v


class Model(nn.Module):
    def __init__(self, ect_size, hidden_size, num_dims, num_pts):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ect_size**2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_dims * num_pts),
        )

    def forward(self, x):
        return self.layer(x)


class BaseModel(L.LightningModule):
    def __init__(self, layer, ect_size, hidden_size, num_pts, learning_rate, num_dims):
        super().__init__()

        self.learning_rate = learning_rate
        self.layer = layer
        self.num_dims = num_dims
        self.num_pts = num_pts

        self.loss_layer = EctLayer(
            EctConfig(
                bump_steps=64,
                num_thetas=64,
                device="cuda:0",
                ect_type="points_derivative",
            ),
            v=generate_uniform_directions(num_thetas=64).cuda(),
        )

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

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
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

        # ect_hat = self.layer(_batch, _batch.batch)
        # print("NUM", self.num_dims)
        # if self.num_dims == 2:
        #     # loss = self.loss_fn(ect_hat, ect)
        #     loss_ch = chamfer_distance(
        #         torch.cat(
        #             [
        #                 batch.x.view(batch_len, 128, self.num_dims),
        #                 torch.zeros(size=(batch_len, 128, 1), device=self.device),
        #             ],
        #             dim=-1,
        #         ),
        #         torch.cat(
        #             [
        #                 _batch.x.view(-1, self.num_pts, self.num_dims),
        #                 torch.zeros(size=(batch_len, 128, 1), device=self.device),
        #             ],
        #             dim=-1,
        #         ),
        #     ).mean()
        # else:
        loss_ch = chamfer_distance(
            batch.x.view(batch_len, -1, self.num_dims),
            _batch.x.view(-1, self.num_pts, self.num_dims),
        ).mean()

        # loss_emd = EMD(
        #     batch.x.view(batch_len, -1, self.num_dims),
        #     _batch.x.view(-1, self.num_pts, self.num_dims),
        #     transpose=False
        # ).mean()

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
        self.loss_layer.v = generate_uniform_directions(d=3, num_thetas=64).cuda()
        return super().on_train_epoch_end()

    # def log_accuracies(
    #     self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    # ):
    #     if step == "train":
    #         accuracy_fn = self.training_accuracy
    #         name = "train_accuracy"
    #     elif step == "test":
    #         accuracy_fn = self.test_accuracy
    #         name = "test_accuracy"
    #     elif step == "validation":
    #         accuracy_fn = self.validation_accuracy
    #         name = "validation_accuracy"
    #
    #     accuracy = accuracy_fn(
    #         x_hat.reshape(batch_len, -1),
    #         y.reshape(batch_len, -1),
    #     )
    #     self.log(
    #         name,
    #         accuracy,
    #         prog_bar=True,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=batch_len,
    #     )

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    # def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
    #     return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")
