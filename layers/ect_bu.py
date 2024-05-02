import torch
import torch.nn as nn
import geotorch
from layers.config import EctConfig
from torch_geometric.data import Data

from typing import Protocol
from dataclasses import dataclass

def compute_wecc(nh, index, lin, dim_size, out, weight=None):
    # print(weight.shape)
    # print(ecc.shape)
    ecc = torch.nn.functional.sigmoid(200 * torch.sub(lin, nh)) * weight.view(1,-1,1)
    res = torch.index_add(out,1, index, ecc).movedim(0, 1)
    return res

def compute_ecc(nh, index, lin, dim_size, out):
    ecc = torch.nn.functional.sigmoid(200 * torch.sub(lin, nh))
    return torch.index_add(out,1, index, ecc).movedim(0, 1)


def compute_ect_points(data, v, lin, out):
    nh = data.x @ v
    return compute_ecc(nh, data.batch, lin, data.num_graphs,out)

def compute_ect_edges(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin, data.num_graphs,out)
        - compute_ecc(eh, data.batch[data.edge_index[0,:]], lin, data.num_graphs,out)
    )

def compute_wect_edges(data, v, lin, out):
    # print("out",out.shape)
    # print('lin',lin.shape)
    nh = data.x @ v
    # if data.edge_index is not None:
    eh, _ = nh[data.edge_index].max(dim=0)
    # print("nh",nh.shape)
    # print("eh",eh.shape)
    # print("data.batch",data.batch.shape)
    return compute_ecc(nh, data.batch, lin, data.num_graphs,out) - compute_wecc(eh, data.batch[data.edge_index[0,:]], lin, data.num_graphs,out,weight=data.weight)
    # else:
    #     return compute_ecc(nh, data.batch, lin, data.num_graphs,out) 


def compute_ect_faces(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin, data.num_graphs,out)
        - compute_ecc(eh, data.batch[data.edge_index[0]], lin, data.num_graphs,out)
        + compute_ecc(fh, data.batch[data.face[0]], lin, data.num_graphs,out)
    )


class EctLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config: EctConfig, fixed=False):
        super().__init__()
        self.config = config
        self.fixed = fixed
        self.lin = (
            torch.linspace(-config.R, config.R, config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if self.fixed:
            self.v = torch.vstack(
                [
                    torch.sin(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                    torch.cos(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                ]
            ).to(config.device)
        else:
            self.v = torch.nn.Parameter(
                torch.rand(size=(config.num_features, config.num_thetas)) - 0.5
            ).to(config.device)

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "w_edges":
            self.compute_ect = compute_wect_edges
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ect_type == "faces":
            self.compute_ect = compute_ect_faces

    def __post_init__(self):
        if self.fixed:
            geotorch.constraints.sphere(self, "v")

    def forward(self, data):
        out = torch.zeros(size=(self.config.bump_steps,data.batch.max()+1,self.config.num_thetas), device=self.config.device)
        return self.compute_ect(data, self.v, self.lin,out)
