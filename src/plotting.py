import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from torch import Tensor

pv.set_jupyter_backend("static")

DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 5, "c": ".5"}
LIGHTRED = [255, 100, 100]


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def plot_recon_2d(recon_pcs, ref_pcs, jnt_pcs=None, num_pc=5):

    if jnt_pcs is None:
        jnt_pcs = np.hstack([ref_pcs, recon_pcs])
        colors = np.array(ref_pcs.shape[1] * ["blue"] + recon_pcs.shape[1] * ["red"])
    else:
        colors = np.array(jnt_pcs.shape[1] * [0.5])

    fig, axes = plt.subplots(nrows=3, ncols=num_pc, figsize=(num_pc * 2, 3 * 2))

    for recon_pc, ref_pc, jnt_pc, axis in zip(recon_pcs, ref_pcs, jnt_pcs, axes.T):
        recon_pc = rotate(recon_pc.reshape(-1, 2), degrees=-90)
        ref_pc = rotate(ref_pc.reshape(-1, 2), degrees=-90)
        jnt_pc = rotate(jnt_pc.reshape(-1, 2), degrees=-90)

        ax = axis[0]
        ax.scatter(recon_pc[:, 0], recon_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")

        ax = axis[1]
        ax.scatter(ref_pc[:, 0], ref_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")

        ax = axis[2]
        ax.scatter(jnt_pc[:, 0], jnt_pc[:, 1], s=5, c=colors)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")
    return fig


def plot_recon_3d(
    recon_pcs,
    ref_pcs=None,
    num_pc=5,
    offset=0,
    filename=None,
    point_size=2,
):

    if type(recon_pcs) == Tensor:
        recon_pcs = recon_pcs.cpu().detach().numpy()

    if type(ref_pcs) == Tensor:
        ref_pcs = ref_pcs.cpu().detach().numpy()

    off_screen = True if filename is not None else False

    pl = pv.Plotter(
        shape=(3, num_pc),
        window_size=[200 * num_pc, 600],
        border=False,
        polygon_smoothing=True,
        off_screen=off_screen,
    )

    for col in range(num_pc):
        # First plat
        pl.subplot(0, col)
        actor = pl.add_points(
            recon_pcs[col + offset].reshape(-1, 3),
            style="points",
            emissive=False,
            show_scalar_bar=False,
            render_points_as_spheres=True,
            color="lightgray",
            point_size=point_size,
            ambient=0.2,
            diffuse=0.8,
            specular=0.8,
            specular_power=40,
            smooth_shading=True,
        )
        pl.subplot(1, col)
        actor = pl.add_points(
            ref_pcs[col + offset].reshape(-1, 3),
            style="points",
            emissive=False,
            show_scalar_bar=False,
            render_points_as_spheres=True,
            color=LIGHTRED,
            point_size=point_size,
            ambient=0.2,
            diffuse=0.8,
            specular=0.8,
            specular_power=40,
            smooth_shading=True,
        )
        actor = pl.add_points(
            recon_pcs[col + offset].reshape(-1, 3),
            style="points",
            emissive=False,
            show_scalar_bar=False,
            render_points_as_spheres=True,
            color="lightgray",
            point_size=point_size,
            ambient=0.2,
            diffuse=0.8,
            specular=0.8,
            specular_power=40,
            smooth_shading=True,
        )

        pl.subplot(2, col)
        actor = pl.add_points(
            ref_pcs[col + offset].reshape(-1, 3),
            style="points",
            emissive=False,
            show_scalar_bar=False,
            render_points_as_spheres=True,
            color=LIGHTRED,
            point_size=2,
            ambient=0.2,
            diffuse=0.8,
            specular=0.8,
            specular_power=40,
            smooth_shading=True,
        )

    pl.background_color = "w"
    pl.link_views()
    pl.camera_position = "xy"
    pos = pl.camera.position
    pl.camera.position = (pos[0], pos[1] + 3, pos[2])
    pl.camera.position = (10, 0, 0)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    # create a top down light
    light = pv.Light(
        position=(0, 0, 0), positional=True, cone_angle=50, exponent=20, intensity=0.2
    )
    pl.add_light(light)
    pl.camera.zoom(0.7)
    if filename is not None:
        pl.show(screenshot=filename)
    else:
        pl.show()


def plot_grid(pcs, num_pc=5):
    pl = pv.Plotter(
        shape=(num_pc, num_pc),
        window_size=[200 * num_pc, 200 * num_pc],
        border=False,
        polygon_smoothing=True,
    )

    for col in range(num_pc):
        for row in range(num_pc):
            # First plat
            pl.subplot(row, col)
            pl.add_points(
                pcs[col + num_pc * row].reshape(-1, 3),
                style="points",
                emissive=False,
                show_scalar_bar=False,
                render_points_as_spheres=True,
                color="lightgray",
                point_size=2,
                ambient=0.2,
                diffuse=0.8,
                specular=0.8,
                specular_power=40,
                smooth_shading=True,
            )

    pl.background_color = "w"
    pl.link_views()
    pl.camera_position = "xy"
    pos = pl.camera.position
    pl.camera.position = (pos[0], pos[1] + 3, pos[2])
    pl.camera.position = (10, 0, 0)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    light = pv.Light(
        position=(0, 0, 0), positional=True, cone_angle=50, exponent=20, intensity=0.2
    )
    pl.add_light(light)
    pl.camera.zoom(0.7)
    pl.show()


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv
import torch.nn as nn
import torch.nn.functional as F


def plot_graph(x, edge_index, edge_weigths=None, ax=None):

    nodes = [i for i in range(len(x))]
    pos_dict = {i: p for i, p in zip(nodes, x)}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_index)
    nx.draw_networkx_nodes(nodes, pos=pos_dict, node_size=100, ax=ax)
    for idx, edge in enumerate(edge_index):
        if edge_weigths is not None:
            if edge_weigths[idx] > 0.01:
                nx.draw_networkx_edges(
                    G,
                    pos_dict,
                    [edge],
                    alpha=edge_weigths[idx],
                    width=2,
                    edge_color="b",
                    ax=ax,
                )
        else:
            nx.draw_networkx_edges(
                G,
                pos_dict,
                [edge],
                width=2,
                edge_color="b",
                ax=ax,
            )
    nx.draw_networkx_labels(G, pos_dict, ax=ax)
    ax.set_aspect(1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    return ax


def plot_ect(ect_gt, ect_pred, num_ects=5, filename=None):

    fig, axes = plt.subplots(nrows=2, ncols=num_ects, figsize=(3 * num_ects, 6))
    for ax, gt, pred in zip(axes.T, ect_gt, ect_pred):

        ax[0].imshow(pred.cpu().detach().squeeze().numpy())
        ax[0].axis("off")

        ax[1].imshow(gt.cpu().squeeze().numpy())
        ax[1].axis("off")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_epoch_ect(x, x_gt, layer_truth, ect_pred, ect_truth):

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    # Plot predicted graph
    ax = axes[0][0]
    ax.set_title("Prediction")
    x_pred = np.round(F.tanh(layer_pred.x).detach().numpy(), decimals=2)
    edge_index_pred = layer_pred.ei.T.detach().numpy()
    edge_weights_pred = nn.functional.sigmoid(layer_pred.ew.detach()).numpy()
    plot_graph(x_pred, edge_index_pred, edge_weights_pred, ax)

    # Plot ground truth graph
    ax = axes[0][1]
    ax.set_title("Ground Truth")

    x_gt = np.round(layer_truth.x.detach().numpy(), decimals=2)
    edge_index_gt = layer_truth.ei.T.detach().numpy()
    plot_graph(x_gt, edge_index_gt, None, ax)

    ax = axes[1][0]
    ax.imshow(ect_pred[0].detach().squeeze().numpy())
    ax.axis("off")
    ax.set_title("Points")

    ax = axes[1][1]
    ax.imshow(ect_truth[0].squeeze().numpy())
    ax.axis("off")
    ax.set_title("Points")

    ax = axes[2][0]
    ax.imshow(ect_pred[1].detach().squeeze().numpy())
    ax.axis("off")
    ax.set_title("Edges")

    ax = axes[2][1]
    ax.imshow(ect_truth[1].squeeze().numpy())
    ax.axis("off")
    ax.set_title("Edges")

    plt.tight_layout()
    plt.savefig(f"./anim/{epoch//10}.png")


def plot_epoch(x, x_gt, epoch):
    pl = pv.Plotter(shape=(1, 3), window_size=[1200, 400], off_screen=True)
    points = x_gt.detach().cpu().view(-1, 3).numpy()
    pl.subplot(0, 0)
    actor = pl.add_points(
        points,
        style="points",
        emissive=False,
        show_scalar_bar=False,
        render_points_as_spheres=True,
        scalars=points[:, 2],
        point_size=5,
        ambient=0.2,
        diffuse=0.8,
        specular=0.8,
        specular_power=40,
        smooth_shading=True,
    )
    pl.subplot(0, 1)
    actor = pl.add_points(
        x_gt.detach().cpu().view(-1, 3).numpy(),
        style="points",
        emissive=False,
        show_scalar_bar=False,
        render_points_as_spheres=True,
        color="lightblue",
        point_size=5,
        ambient=0.2,
        diffuse=0.8,
        specular=0.8,
        specular_power=40,
        smooth_shading=True,
    )
    points = x.reshape(-1, 3).detach().cpu().numpy()
    actor = pl.add_points(
        points,
        style="points",
        emissive=False,
        show_scalar_bar=False,
        render_points_as_spheres=True,
        color="red",
        point_size=5,
        ambient=0.2,
        diffuse=0.8,
        specular=0.8,
        specular_power=40,
        smooth_shading=True,
    )
    pl.subplot(0, 2)
    points = x.reshape(-1, 3).detach().cpu().numpy()
    actor = pl.add_points(
        points,
        style="points",
        emissive=False,
        show_scalar_bar=False,
        render_points_as_spheres=True,
        scalars=points[:, 2],
        point_size=5,
        ambient=0.2,
        diffuse=0.8,
        specular=0.8,
        specular_power=40,
        smooth_shading=True,
    )

    pl.background_color = "w"
    pl.link_views()
    pl.camera_position = "xy"
    pos = pl.camera.position
    pl.camera.position = (pos[0] + 3, pos[1], pos[2])
    pl.camera.azimuth = 145
    pl.camera.elevation = 20
    # create a top down light
    light = pv.Light(
        position=(0, 0, 3), positional=True, cone_angle=50, exponent=20, intensity=0.2
    )
    pl.add_light(light)
    pl.camera.zoom(0.7)
    pl.screenshot(f"./img/{epoch}.png")
