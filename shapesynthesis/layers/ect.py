"""
Ect layer implementation
"""

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import torch
from pydantic import BaseModel
from torch import nn
from torch_geometric.data import Batch

Tensor: TypeAlias = torch.Tensor


@dataclass
class EctBatch(Batch):
    x: Tensor | None = None
    ect: Tensor | None = None


class EctConfig(BaseModel):
    """
    Config for initializing an ect layer.
    """

    num_thetas: int
    resolution: int
    r: float
    scale: float
    ect_type: Literal["points"]
    ambient_dimension: int
    normalized: bool
    seed: int


def compute_ect_point_cloud(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
) -> Tensor:
    """
    Computes the ECT of a point cloud.

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into. (Is irrelevant for
        this experiment.)
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multipicative factor for the sigmoid function.

    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,N,R] where B is the number of
        point clouds (thus ECT's), N is the number of direction and R is the
        resolution.
    """
    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return ect


def compute_ecc(nh, index, lin, scale):
    """
    Computes the ECC of a set of points given the node heights.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    out = torch.zeros(
        size=(
            ecc.shape[0],
            index.max().item() + 1,
            ecc.shape[2],
        ),
        device=nh.device,
    )
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


#@torch.compile
def compute_ect_points(x, index, v, lin, scale):
    """Compute the ECT of a set of points."""
    nh = x @ v
    return compute_ecc(nh, index, lin, scale)


class EctLayer(nn.Module):
    """Docstring for EctLayer."""

    def __init__(self, config: EctConfig, v: Tensor):

        super().__init__()
        self.config = config

        # Make the resolution grid.
        self.lin = torch.nn.Parameter(
            torch.linspace(-config.r, config.r, config.resolution).view(-1, 1, 1),
            requires_grad=False,
        )

        self.v = v

    def forward(self, batch: EctBatch, index):
        """Forward method"""
        ect = compute_ect_points(batch.x, index, self.v, self.lin, self.config.scale)
        if self.config.normalized:
            return ect / torch.amax(ect, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        return ect
