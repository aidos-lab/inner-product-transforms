import functools
import operator
from typing import TypeAlias

import lightning as L
import pydantic
import torch
from torch import nn

from layers.ect import EctConfig

Tensor: TypeAlias = torch.Tensor


class ModelConfig(pydantic.BaseModel):
    module: str
    num_pts: int
    learning_rate: float
    ectlossconfig: EctConfig
    ectconfig: EctConfig


class Model(nn.Module):
    """
    The core model that reconstructs an ECT back into a point cloud.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = nn.Sequential(
            ###########################################################
            nn.Conv1d(
                config.ectconfig.num_thetas,
                2 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                4 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=4 * config.ectconfig.num_thetas),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                4 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=8 * config.ectconfig.num_thetas),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                8 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
            ),
        )

        # Function to calculate the shape of the CNN output, in order to
        # initialize the linear layers without having to adjust the input
        # dimension manually.
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

        # Ambient dimension is 3 for 3D and 2 for 2D point clouds.
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
        """
        We compute the forward pass here. The input ECT is viewed as a image and
        each pixel has values between [0,1]. We rescale to [-1,1] to accommodat
        the CNN layers, who prefer this type of input. Lastly, the Tanh
        activation function at the end ensures that the models output is
        relatively bounded.
        """
        ect = ect.squeeze().movedim(-1, -2)
        x = self.conv(ect)
        x = self.layer(x.flatten(start_dim=1))
        return self.config.ectconfig.r * x.view(
            -1, self.config.num_pts, self.config.ectconfig.ambient_dimension
        )
