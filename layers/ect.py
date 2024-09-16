"""
Ect layer implementation
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import Data


@dataclass(frozen=True)
class EctConfig:
    """
    Config for initializing an ect layer.
    """

    num_thetas: int = 32
    bump_steps: int = 32
    r: float = 1.1
    ect_type: str = "points"
    device: str = "cuda:0"
    num_features: int = 3
    normalized: bool = False


def compute_ecc_derivative(nh, index, lin, out, scale):
    """
    Computes the ECC with the derivative of the sigmoid instead of the
    sigmoid.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    )
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ecc(nh, index, lin, out, scale):
    """
    Computes the ECC of a set of points given the node heights.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ect_points_derivative(data, index, v, lin, out, scale):
    """Compute the derivative of the ECT of a set of points."""
    nh = data.x @ v
    return compute_ecc_derivative(nh, index, lin, out, scale)


def compute_ect_points(data, index, v, lin, out, scale):
    """Compute the ECT of a set of points."""
    nh = data.x @ v
    return compute_ecc(nh, index, lin, out, scale)


class EctLayer(nn.Module):
    """Docstring for EctLayer."""

    def __init__(self, config: EctConfig, v=None):
        super().__init__()
        self.config = config
        self.lin = (
            torch.linspace(-config.r, config.r, config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if v is None:
            raise AttributeError("Please provide the directions")
        self.v = v

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "points_derivative":
            self.compute_ect = compute_ect_points_derivative

    def forward(self, data: Data, index, scale=500):
        """Forward method"""
        out = torch.zeros(
            size=(
                self.config.bump_steps,
                index.max().item() + 1,
                self.config.num_thetas,
            ),
            device=self.config.device,
        )
        ect = self.compute_ect(data, index, self.v, self.lin, out, scale)
        if self.config.normalized:
            return ect / torch.amax(ect, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        return ect
