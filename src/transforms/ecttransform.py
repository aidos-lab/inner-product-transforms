"""
All transforms for the datasets.
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import pydantic
import torch
from torch import nn

from src.layers.directions import generate_2d_directions, generate_uniform_directions
from src.layers.ect import EctConfig, compute_ect_point_cloud


class TransformConfig(pydantic.BaseModel):
    module: str
    structured: bool
    num_thetas: int
    resolution: int
    r: float
    scale: float
    ect_type: Literal["points"]
    ambient_dimension: int
    normalized: bool
    seed: int


class Transform(nn.Module):
    def __init__(self, config: TransformConfig):
        super().__init__()
        self.config = config

        if not hasattr(config, "structured"):
            structured = False
        else:
            structured = config.structured

        if structured and config.ambient_dimension == 2:
            v = generate_2d_directions(config.num_thetas)
        else:
            v = generate_uniform_directions(
                config.num_thetas, d=config.ambient_dimension, seed=config.seed
            )

        self.v = torch.nn.Parameter(v, requires_grad=False)

        self.ect_fn = torch.compile(
            partial(
                compute_ect_point_cloud,
                v=self.v,
                radius=self.config.r,
                resolution=self.config.resolution,
                scale=self.config.scale,
            )
        )

    def __call__(self, x):
        return self.ect_fn(x)


class Ect2DTransform:
    def __init__(self, config: EctConfig, device="cpu"):
        self.config = config
        self.v = generate_2d_directions(config.num_thetas).to(device)
        self.ect_fn = torch.compile(
            partial(
                compute_ect_point_cloud,
                v=self.v,
                radius=self.config.r,
                resolution=self.config.resolution,
                scale=self.config.scale,
            )
        )

    def __call__(self, x):
        return self.ect_fn(x)


class RandomSamplePoints:
    """Randomly Choose points"""

    def __call__(self, x):
        idx = np.random.choice(size=2048, replace=False)
        return x[idx]
