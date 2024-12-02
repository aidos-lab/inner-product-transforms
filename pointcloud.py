import torch
from layers.directions import generate_uniform_directions
from layers.ect import compute_ect_point_cloud
import pyvista as pv
from torch import nn
import matplotlib.pyplot as plt

from plotting import plot_epoch

from kaolin.metrics.pointcloud import chamfer_distance

num_epochs = 100
x_init = torch.rand(size=(1, 2048, 3)).cuda()
v = generate_uniform_directions(num_thetas=256, d=3, seed=2024).cuda()
x_gt = torch.load("./results/encoder_chair_sparse/references.pt")[0]
ect_gt = compute_ect_point_cloud(x_gt.view(1, 2048, 3), v, radius=5, resolution=256)


x = nn.Parameter(x_init)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    params=[x],
    lr=0.2,
)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    ect_pred = compute_ect_point_cloud(x, v, radius=5, resolution=256)
    loss = loss_fn(ect_pred, ect_gt)
    loss.backward()
    optimizer.step()
    # print(loss.item())
    # scheduler.step()
    # print(epoch, loss.item())
    # if epoch % 5 == 0:
    #     print(
    #         epoch,
    #         loss.item(),
    #         chamfer_distance(
    #             x.view(-1, , 3), x_gt.view(-1, num_pts, 3)
    #         ).item(),
    #     )
    #     with torch.no_grad():
    #         plot_epoch(x[0], x_gt, epoch)
[xi.detach() for xi in x]
