import torch
from layers.ect import compute_ect_point_cloud
from torch import nn


def render_point_cloud_half(x_init, ect_gt, v, num_epochs, scale, radius, resolution):
    # x = [nn.Parameter(x_in.unsqueeze(0)) for x_in in x_init]
    x = nn.Parameter(x_init)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=[x],
        lr=0.5,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 200, 1000], gamma=0.5
    )

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        ect_pred = compute_ect_point_cloud(
            x, v, radius=radius, resolution=resolution, scale=scale
        )
        loss = loss_fn(ect_pred, ect_gt)
        loss.backward()
        optimizer.step()

        scheduler.step()
    return x.detach()


def render_point_cloud(x_init, ect_gt, v, num_epochs, scale, radius, resolution):
    # x = [nn.Parameter(x_in.unsqueeze(0)) for x_in in x_init]
    x = nn.Parameter(x_init)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=[x],
        lr=0.5,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 200, 1000], gamma=0.5
    )

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        ect_pred = compute_ect_point_cloud(
            x, v, radius=radius, resolution=resolution, scale=scale
        )
        loss = loss_fn(ect_pred, ect_gt)
        loss.backward()
        optimizer.step()

        scheduler.step()
    return x.detach()


def render_point_cloud_viz(x_init, ect_gt, v, num_epochs, scale, radius, resolution):
    # x = [nn.Parameter(x_in.unsqueeze(0)) for x_in in x_init]
    result = [x_init.clone()]
    x = nn.Parameter(x_init)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=[x],
        lr=0.5,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 200, 1000], gamma=0.5
    )

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        ect_pred = compute_ect_point_cloud(
            x, v, radius=radius, resolution=resolution, scale=scale
        )
        loss = loss_fn(ect_pred, ect_gt)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            result.append(x.detach().clone())

        scheduler.step()
    return x.detach(), torch.cat(result, dim=0)
