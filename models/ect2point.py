import torch
from torch import nn

from typing import TypeAlias

# from torch import tensor as Tensor

Tensor: TypeAlias = torch.Tensor


class Autoencoder(nn.Module):
    def __init__(self, num_pts: int, ect_size: int, hidden_size: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ect_size**2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * num_pts),
        )

    def forward(self, ect: Tensor) -> Tensor:
        return self.encoder(ect)
