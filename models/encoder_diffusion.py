from typing import TypeAlias, Literal
import torch
from torch import nn
import lightning as L
from layers.directions import generate_directions

from kaolin.metrics.pointcloud import chamfer_distance

from torchmetrics.regression import MeanSquaredError


Tensor: TypeAlias = torch.Tensor

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.Conv2d(8,16,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=3),
        )

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*2*32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        return x

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
        batch_len = len(batch.y)

        _batch = batch.clone()
        x_noisy = 0.5 * torch.rand_like(_batch.x).cuda() 
        x_hat = x_noisy.view(-1,128,2).repeat_interleave(128,dim=0)
        _batch.x = (x_hat - x_noisy.unsqueeze(1)).view(-1,2)

        # print(_batch.x.shape)

        ect = self.layer(_batch, torch.arange(128*8).repeat_interleave(128).cuda()).squeeze().unsqueeze(1)
        # print("ECT XXXXXXXXXXXXXXX", ect.shape)
        # print(batch.x.shape)

        x_diff =  self(ect)
        # print(x_diff.shape)
        x_pred = x_noisy + x_diff

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
                        x_pred.view(-1,128,2),
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
        # self.log_accuracies( batch_len, step)
        return loss

    # def log_accuracies(
    #     self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    # ):
        # if step == "train":
        #     accuracy_fn = self.training_accuracy
        #     name = "train_accuracy"
        # elif step == "test":
        #     accuracy_fn = self.test_accuracy
        #     name = "test_accuracy"
        # elif step == "validation":
        #     accuracy_fn = self.validation_accuracy
        #     name = "validation_accuracy"
        #
        # accuracy = accuracy_fn(
        #     x_hat.reshape(batch_len, -1),
        #     y.reshape(batch_len, -1),
        # )
        # self.log(
        #     name,
        #     accuracy,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=batch_len,
        # )

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    # def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
    #     return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")
