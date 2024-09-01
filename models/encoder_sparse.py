from typing import TypeAlias, Literal
import torch
from torch import nn
import lightning as L
from layers.directions import generate_directions

from kaolin.metrics.pointcloud import chamfer_distance

from torchmetrics.regression import MeanSquaredError


Tensor: TypeAlias = torch.Tensor


# class Model(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#         # self.layer = nn.Sequential(
#         #     nn.Flatten(),
#         #     nn.Linear(64**2, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 256),
#         # )
#
#         self.conv = nn.Sequential(
#             nn.Conv1d(1,512,kernel_size=64,stride=64),
#             nn.ReLU(),
#         )
#
#         self.layer = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#         )
#
#     def forward(self, x):
#         x = x.flatten(start_dim=1).unsqueeze(1)
#         x = self.conv(x).max(axis=-2)[0]
#         x = self.layer(x)
#         print("XXXXXXXXXXXXXXXXXXXXXXXXXXX",x.shape)
#         return x


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv1d(1,128,kernel_size=128,stride=128),
            nn.ReLU(),
        )

        # self.conv = nn.Sequential(
        #     nn.ConvTranspose1d(1,64,3),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(64,128,3),
        # )

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )


    def forward(self, x):
        x = self.conv(x.flatten(start_dim=1).unsqueeze(1)).max(axis=-2)[0]
        x = self.layer(x)
        # x = self.layer(x.movedim(-1,-2)).sum(axis=-2) # .max(axis=-2)[0]
        return x



# class Model(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#         self.layer = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#         )
#
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x
#

# def inv(w):
#     w1 = w.T
#     w2 = (w @ w.T).inverse()
#     return w1 @ w2
#
# class Model(nn.Module):
#     def __init__(self, v) -> None:
#         super().__init__()
#
#         # self.layer = nn.Sequential(
#         #     nn.Flatten(),
#         #     nn.Linear(64**2, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 256),
#         # )
#         self.v_inv = inv(v)
#
#         self.conv = nn.Sequential(
#             nn.Conv1d(1,128,kernel_size=64,stride=64),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = x.flatten(start_dim=1).unsqueeze(1)
#         x = self.conv(x)
#         x = x @ self.v_inv
#         return x




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
        self.model = Model()
        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64**2, 52),
        #     nn.ReLU(),
        #     nn.Linear(52, 52),
        #     nn.ReLU(),
        #     nn.Linear(52, 52),
        #     nn.ReLU(),
        #     nn.Linear(52, 256),
        # )

        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(ect_size**2, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_dims * num_pts),
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
        batch_len = len(batch.y)
        _batch = batch.clone()

        ect = self.layer(batch, batch.batch)

        _batch.x = self(ect.movedim(-1, -2)).reshape(-1, self.num_dims)

        _batch.batch = torch.arange(
            batch.batch.max().item() + 1, device=self.device
        ).repeat_interleave(self.num_pts)

        ect_hat = self.layer(_batch, _batch.batch)
        # print("NUM", self.num_dims)
        if self.num_dims == 2:
            # loss = self.loss_fn(ect_hat, ect)
            loss_ch = chamfer_distance(
                torch.cat(
                    [
                        batch.x.view(batch_len, 128, self.num_dims),
                        torch.zeros(size=(batch_len, 128, 1), device=self.device),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        _batch.x.view(-1, self.num_pts, self.num_dims),
                        torch.zeros(size=(batch_len, 128, 1), device=self.device),
                    ],
                    dim=-1,
                ),
            ).mean()
        else:
            loss_ch = chamfer_distance(
                batch.x.view(batch_len, -1, self.num_dims),
                _batch.x.view(-1, self.num_pts, self.num_dims),
            ).mean()

        loss = loss_ch
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

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    # def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
    #     return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")
