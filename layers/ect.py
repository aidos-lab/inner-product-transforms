import torch
import torch.nn as nn
import geotorch
from torch_geometric.data import Data, Batch

from typing import Protocol
from dataclasses import dataclass


@dataclass(frozen=True)
class EctConfig:
    num_thetas: int = 32
    bump_steps: int = 32
    R: float = 1.1
    ect_type: str = "points"
    device: str = "cuda:0"
    num_features: int = 3
    normalized: bool = False


def compute_ecc_derivative(nh, index, lin, out):
    ecc = torch.nn.functional.sigmoid(50 * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(50 * torch.sub(lin, nh))
    )
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ecc(nh, index, lin, out):
    ecc = torch.nn.functional.sigmoid(500 * torch.sub(lin, nh))
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ect_points_derivative(data, index, v, lin, out):
    nh = data.x @ v
    return compute_ecc_derivative(nh, index, lin, out)


def compute_ect_points(data, index, v, lin, out):
    nh = data.x @ v
    return compute_ecc(nh, index, lin, out)


def compute_ect_edges(data, index, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    return compute_ecc(nh, index, lin, out) - compute_ecc(
        eh, index[data.edge_index[0]], lin, out
    )


def compute_ect_faces(data, index, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, index, lin, out)
        - compute_ecc(eh, index[data.edge_index[0]], lin, out)
        + compute_ecc(fh, index[data.face[0]], lin, out)
    )


class EctLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config: EctConfig, v=None):
        super().__init__()
        self.config = config
        self.lin = (
            torch.linspace(-config.R, config.R, config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if v is None:
            raise AttributeError("Please provide the directions")
        self.v = v

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ect_type == "faces":
            self.compute_ect = compute_ect_faces
        elif config.ect_type == "points_derivative":
            self.compute_ect = compute_ect_points_derivative

    def forward(self, data, index):
        out = torch.zeros(
            size=(
                self.config.bump_steps,
                index.max().item() + 1,
                self.config.num_thetas,
            ),
            device=self.config.device,
        )
        ect = self.compute_ect(data, index, self.v, self.lin, out)
        if self.config.normalized:
            return ect / torch.amax(ect, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        else:
            return ect


if __name__ == "__main__":
    NUM_THETAS = 64
    DEVICE = "cuda:0"
    V = torch.vstack(
        [
            torch.sin(
                torch.linspace(0, 2 * torch.pi, NUM_THETAS, device=DEVICE)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, NUM_THETAS, device=DEVICE)
            ),
        ]
    )
    layer = EctLayer(
        EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS), v=V
    ).to(DEVICE)

    data = Data(x=torch.rand(size=(4, 2))).to(DEVICE)
    index = torch.tensor([0, 0, 1, 1]).to(DEVICE)

    layer(data, index)
