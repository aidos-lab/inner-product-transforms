"""
All transforms for the datasets.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from shapesynthesis.layers.directions import (
    generate_2d_directions,
    generate_uniform_directions,
)
from shapesynthesis.layers.ect import EctConfig, compute_ect_point_cloud


@dataclass
class EctTransformConfig(EctConfig):
    structured: bool


class EctTransform:
    def __init__(self, config: EctTransformConfig, device="cpu"):
        self.config = config

        if not hasattr(config, "structured"):
            structured = False
        else:
            structured = config.structured

        if structured and config.ambient_dimension == 2:
            self.v = generate_2d_directions(config.num_thetas).to(device)
        else:
            self.v = generate_uniform_directions(
                config.num_thetas, d=config.ambient_dimension, seed=config.seed
            ).to(device)
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


class RandomRotate:
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def __call__(self, x):
        # Max 45 degree rotation per axis
        angles = 0.15 * torch.pi * torch.rand(size=(len(x),))
        sin, cos = torch.sin(angles), torch.cos(angles)
        vec_1 = torch.ones(size=(len(x),))
        vec_0 = torch.zeros(size=(len(x),))

        if self.axis == 0:
            matrix = torch.stack(
                [
                    torch.stack([vec_1, vec_0, vec_0]),
                    torch.stack([vec_0, cos, sin]),
                    torch.stack([vec_0, -sin, cos]),
                ]
            )
        elif self.axis == 1:
            matrix = torch.stack(
                [
                    torch.stack([cos, vec_0, -sin]),
                    torch.stack([vec_0, vec_1, vec_0]),
                    torch.stack([sin, vec_0, cos]),
                ]
            )
        else:
            matrix = torch.stack(
                [
                    torch.stack([cos, sin, vec_0]),
                    torch.stack([-sin, cos, vec_0]),
                    torch.stack([vec_0, vec_0, vec_1]),
                ]
            )

        return torch.bmm(x, matrix.movedim(0, -1).movedim(0, -2).cuda())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"
