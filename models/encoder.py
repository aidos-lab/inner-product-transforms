"""Encoder module"""

from typing import Literal
import torch
from torch import nn
import lightning as L
from kaolin.metrics.pointcloud import chamfer_distance
from layers.ect import EctLayer
from layers.directions import generate_directions, generate_uniform_directions


class BaseModel(L.LightningModule):
    def __init__(self, ectconfig, ectlossconfig, modelconfig):
        super().__init__()
        self.ectconfig = ectconfig
        self.ectlossconfig = ectlossconfig
        self.modelconfig = modelconfig

        self.visualization = []
        self.validation_visualization = []

        self.save_hyperparameters()

        self.layer = EctLayer(
            ectconfig,
            v=generate_directions(
                ectconfig.num_thetas, modelconfig.num_dims
            ).cuda(),
        )

        self.loss_layer = EctLayer(
            ectlossconfig,
            v=generate_uniform_directions(
                num_thetas=ectlossconfig.num_thetas
            ).cuda(),
        )

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                ectconfig.num_thetas * ectconfig.bump_steps,
                modelconfig.hidden_size,
            ),
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
            self.model.parameters(), lr=self.modelconfig.learning_rate
        )
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(
        self, batch, batch_idx, step: Literal["train", "test", "validation"]
    ):
        batch_len = len(batch)
        _batch = batch.clone()

        # Compute ECT
        ect = self.layer(_batch, _batch.batch)

        # Reconstruct the batch
        _batch.x = self(ect).view(-1, self.modelconfig.num_dims)

        if self.modelconfig.num_dims == 2:
            loss_cd = chamfer_distance(
                torch.cat(
                    [
                        batch.x.view(batch_len, 128, 2),
                        torch.zeros(
                            size=(batch_len, 128, 1), device=self.device
                        ),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        _batch.x.view(-1, 128, 2),
                        torch.zeros(
                            size=(batch_len, 128, 1), device=self.device
                        ),
                    ],
                    dim=-1,
                ),
            ).mean()
        else:
            loss_cd = chamfer_distance(
                batch.x.view(batch_len, -1, self.modelconfig.num_dims),
                _batch.x.view(
                    -1, self.modelconfig.num_pts, self.modelconfig.num_dims
                ),
            ).mean()
        # CD Loss

        # ect_pred = self.loss_layer(_batch, _batch.batch)
        # ect_target = self.loss_layer(batch, batch.batch)

        # eps = 10e-5
        # ect_pred /= ect_pred.sum(axis=1, keepdim=True)
        # ect_target /= ect_target.sum(axis=1, keepdim=True)
        # ect_pred += eps
        # ect_target += eps

        # ect_kld_loss = (
        #     F.kl_div(ect_pred.log(), ect_target, None, None, reduction="none")
        #     .sum(dim=-1)
        #     .sum(dim=-1)
        #     / self.modelconfig.num_pts
        # ).mean()

        loss = loss_cd
        ect_kld_loss = 0

        self.log_dict(
            {
                f"{step}_loss": loss,
                f"{step}_kld_loss": ect_kld_loss,
                f"{step}_loss_cd": loss_cd,
            },
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        # if batch_idx == 0:
        #     self.visualization = [
        #         (
        #             batch.x.view(
        #                 -1, self.modelconfig.num_pts, self.modelconfig.num_dims
        #             ),
        #             _batch.x.view(
        #                 -1, self.modelconfig.num_pts, self.modelconfig.num_dims
        #             ),
        #         )
        #     ]
        #
        # if batch_idx == 0 and step == "validation":
        #     self.validation_visualization = [
        #         (
        #             batch.x.view(
        #                 -1, self.modelconfig.num_pts, self.modelconfig.num_dims
        #             ),
        #             _batch.x.view(
        #                 -1, self.modelconfig.num_pts, self.modelconfig.num_dims
        #             ),
        #         )
        #     ]
        return loss

    # def on_train_epoch_end(self) -> None:
    #     self.loss_layer.v = generate_uniform_directions(
    #             num_thetas=self.ectlossconfig.num_thetas
    #         ).cuda()
    #     return super().on_train_epoch_end()

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        with torch.no_grad():
            loss = self.general_step(batch, batch_idx, "validation")

        # if self.current_epoch % 10 == 0 and batch_idx== 0:
        #     tensorboard_logger = self.logger.experiment
        #     ref_pcs, sample_pcs = self.validation_visualization[0]
        #     fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(16, 6))
        #     fig.tight_layout()
        #     for axis, ref, pred in zip(axes.T, ref_pcs, sample_pcs):
        #         pts_ref = ref.view(-1, 2).cpu().detach()
        #         ax = axis[0]
        #         ax.scatter(pts_ref[:, 0], pts_ref[:, 1], s=0.1)
        #         ax.axis("off")

        #         ax = axis[1]
        #         pts_pred = pred.view(-1, 2).cpu().detach()
        #         ax.scatter(pts_pred[:, 0], pts_pred[:, 1], s=0.1)
        #         ax.axis("off")
        #         ax = axis[2]
        #         pts = torch.vstack([pts_pred, pts_ref])
        #         ax.scatter(pts[:, 0], pts[:, 1], s=0.1)
        #         ax.axis("off")

        #     # Adding plot to tensorboard
        #     tensorboard_logger.add_figure(
        #         "val_reconstruction", plt.gcf(), global_step=self.global_step
        #     )

        # self.visualization.clear()
        return loss
