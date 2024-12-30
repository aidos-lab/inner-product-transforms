import functools
import operator
from typing import TypeAlias, Literal
from pydantic import BaseModel
import torch
from torch import nn
import lightning as L
import wandb
from metrics.loss import chamfer2D, chamfer3D

from layers.ect import EctLayer, EctConfig
from layers.directions import generate_uniform_directions
from plotting import plot_recon_2d


Tensor: TypeAlias = torch.Tensor


class ModelConfig(BaseModel):
    """
    Base configuration for the model and contains all parameters. The module
    provides the relative path to the file. Number of points determines the
    number of points that the model outputs. The ECT config is the configuration
    of the ECT used for the input image.
    """

    module: str
    num_pts: int
    learning_rate: float
    ectlossconfig: EctConfig
    ectconfig: EctConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                config.ectconfig.num_thetas,
                kernel_size=3,
                stride=1,
            ),
        )

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
        ect = 2 * ect - 1
        ect = ect.movedim(-1, -2)
        x = self.conv(ect)
        x = self.layer(x.flatten(start_dim=1))
        return x


class BaseLightningModel(L.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.model = Model(config)
        self.losslayer = EctLayer(
            config.ectlossconfig,
            v=generate_uniform_directions(
                config.ectlossconfig.num_thetas,
                d=config.ectlossconfig.ambient_dimension,
                seed=config.ectlossconfig.seed,
            ).cuda(),
        )
        self.count = 0

        if config.ectlossconfig.ambient_dimension == 3:
            self.loss_fn = chamfer3D
        elif config.ectlossconfig.ambient_dimension == 2:
            self.loss_fn = chamfer2D
        else:
            raise ValueError(
                f"Number of dimensions {config.ectlossconfig.ambient_dimension} not supported"
            )

        self.save_hyperparameters()

        self.plot_batch = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def forward(self, ect):
        x = self.model(ect)
        return x

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):

        batch_len = len(batch)
        pc_shape = batch[0].x.shape
        _batch = batch.clone()

        _batch.x = self(batch.ect).reshape(
            -1, self.config.ectlossconfig.ambient_dimension
        )

        _batch.batch = torch.arange(
            batch.batch.max().item() + 1, device=self.device
        ).repeat_interleave(self.config.num_pts)

        loss = self.loss_fn(
            _batch.x.view(
                -1, self.config.num_pts, self.config.ectconfig.ambient_dimension
            ),
            batch.x.view(-1, pc_shape[0], pc_shape[1]),
        )
        self.log_dict(
            {f"{step}_loss": loss},
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        # Save one batch for plotting.
        if batch_idx == 0:
            self.plot_batch = batch

        return self.general_step(batch, batch_idx, "validation")

    # def on_validation_epoch_end(self):
    #     with torch.no_grad():
    #         recon = self.model(self.plot_batch.ect)

    #     fig = plot_recon_2d(
    #         recon_pcs=recon.view(-1, self.config.num_pts, 2).cpu().detach().numpy(),
    #         ref_pcs=self.plot_batch.x.view(-1, 128, 2).cpu().numpy(),
    #         num_pc=10,
    #     )

    #     # # Option 2 for specifically logging images
    #     self.logger.log_image(key="generated_images", images=[fig])
