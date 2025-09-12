"""Experimental version of the encoder."""

import functools
import operator
from typing import TypeAlias

import lightning as L
import pydantic
import torch
from torch import nn

from src.layers.ect import EctConfig

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
                padding=3,
            ),
            nn.BatchNorm1d(num_features=2 * config.ectconfig.num_thetas),
            nn.SiLU(),
            # nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                2 * config.ectconfig.num_thetas,
                4 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=4 * config.ectconfig.num_thetas),
            nn.SiLU(),
            # nn.MaxPool1d(kernel_size=2),
            ###########################################################
            nn.Conv1d(
                4 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=8 * config.ectconfig.num_thetas),
            nn.SiLU(),
            ###########################################################
            nn.Conv1d(
                8 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=8 * config.ectconfig.num_thetas),
            nn.SiLU(),
            ###########################################################
            nn.Conv1d(
                8 * config.ectconfig.num_thetas,
                8 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=8 * config.ectconfig.num_thetas),
            nn.SiLU(),
            ###########################################################
            nn.Conv1d(
                8 * config.ectconfig.num_thetas,
                16 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=16 * config.ectconfig.num_thetas),
            nn.SiLU(),
            ###########################################################
            nn.Conv1d(
                16 * config.ectconfig.num_thetas,
                3 * 16 * config.ectconfig.num_thetas,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )

        # Input size: [64, 6144, 8]
        self.layer = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.ectconfig.ambient_dimension),
        )

    def forward(self, ect):
        """
        We compute the forward pass here. The input ECT is viewed as a image and
        each pixel has values between [0,1]. We rescale to [-1,1] to accommodat
        the CNN layers, who prefer this type of input. Lastly, the Tanh
        activation function at the end ensures that the models output is
        relatively bounded.
        """
        ect = ect.movedim(-1, -2)

        # Output size: [64, 6144, 8]
        x = self.conv(ect)
        x = self.layer(x)
        return x
