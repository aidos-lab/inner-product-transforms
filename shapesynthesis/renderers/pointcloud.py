import torch
from layers.directions import generate_uniform_directions
from layers.ect import compute_ect_point_cloud
import pyvista as pv
from torch import nn
import matplotlib.pyplot as plt

from plotting import plot_epoch
from kaolin.metrics.pointcloud import chamfer_distance


def render_point_cloud(
    x_init,
    ect_gt,
    v,
    num_pts,
    num_epochs,
    x_gt,
    scale,
    radius,
):
    x = nn.Parameter(x_init)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=[x],
        lr=1,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50, 200], gamma=0.5
    )
    plot_epoch(x[0], x_gt, 0)
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        ect_pred = compute_ect_point_cloud(
            x, v, radius=radius, resolution=256, scale=scale
        )
        loss = loss_fn(ect_pred, ect_gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 5 == 0:
            print(
                epoch,
                loss.item(),
                chamfer_distance(
                    x.view(-1, num_pts, 3), x_gt.view(-1, num_pts, 3)
                ).item(),
            )
        #     with torch.no_grad():
        #         plot_epoch(x[0], x_gt, epoch)
    return x.detach()
