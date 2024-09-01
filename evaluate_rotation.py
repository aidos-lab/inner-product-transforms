""" Evaluate rotation """

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

import numpy as np 

from datasets import load_datamodule
from load_models import load_encoder


config = OmegaConf.load("./configs/config_encoder_mnist_rotated.yaml")

dm = load_datamodule(config.data)
model = load_encoder(config)

"""
How are we going to do the whole evaluation:
-  Plot
    - On x axis the directions 
    - On the y - axis the average loss per class

for loop, 
loop over element, 
compute the full orbit, 
pass each element through the decoder 
create a similar rotated point cloud 
and compute the loss per elements 

Build dictionary 
for each class one dict 
for each dict the running total loss and number of samples.
"""


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0].mean(axis=1) +  P.min(2)[0].mean(axis=1)


def rotate(p, angle):
    o = torch.tensor([[0.0, 0.0]])
    R = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]
    )
    return torch.squeeze((R @ (p.T - o.T) + o.T).T)


# Second index is the rotation.
loss_dict = {i:torch.zeros(size=(64,)) for i in range(10)}
class_counts = {i:0 for i in range(10)}


def update_loss_dict(data, losses):
    label = data.y
    angles = torch.linspace(0, 2 * torch.pi, 64)
    point_clouds = torch.stack([rotate(data.x, angle).view(128,2) for angle in angles]).cuda()
    batch = Batch.from_data_list([Data(x=pc.view(-1,2)) for pc in point_clouds])
    with torch.no_grad():
        rotated_ect = model.layer(
            batch, batch.batch
        ).unsqueeze(1)
        recon_pc = model.model.forward(rotated_ect)
    ch = distChamfer(point_clouds,recon_pc.view(-1,128,2))
    return [torch.stack([torch.tensor(angle), label[0], ch.cpu()[angle]]) for angle in range(64) ]


res = []
for data in dm.test_ds:
    res.extend(update_loss_dict(data, loss_dict))

res = torch.stack(res)
torch.save(res, "./results/rotation.pt")


