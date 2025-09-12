import pydantic
import torch
from torch import nn


class TransformConfig(pydantic.BaseModel):
    module: str
    axis: int
    max_degrees: float


class Transform(nn.Module):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, config: TransformConfig) -> None:
        super().__init__()
        self.config = config

    def __call__(self, x):
        # Max 45 degree rotation per axis
        angles = (
            (self.config.max_degrees * (torch.rand(size=(len(x),)) * 2 - 1))
            * torch.pi
            / 180
        )
        sin, cos = torch.sin(angles), torch.cos(angles)
        vec_1 = torch.ones(size=(len(x),))
        vec_0 = torch.zeros(size=(len(x),))

        if self.config.axis == 0:
            matrix = torch.stack(
                [
                    torch.stack([vec_1, vec_0, vec_0]),
                    torch.stack([vec_0, cos, sin]),
                    torch.stack([vec_0, -sin, cos]),
                ]
            )
        elif self.config.axis == 1:
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
