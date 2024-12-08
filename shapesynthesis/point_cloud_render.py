import torch
from layers.directions import generate_uniform_directions
import pyvista as pv
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud
from datasets.shapenetcore import DataModule, DataModuleConfig
from layers.ect import EctConfig
from plotting import plot_epoch

NUM_EPOCHS = 1000
RESOLUTION = torch.tensor(128)
SCALE = RESOLUTION * 0.25
IDX = 8
SEED = 2013
RADIUS = torch.tensor(7)
DTYPE = torch.float32
CATE = "airplane"

v = (
    generate_uniform_directions(num_thetas=RESOLUTION, d=3, seed=SEED)
    .type(DTYPE)
    .cuda()
)
# x_gt_pcs = torch.load("./results/encoder_chair_sparse/references.pt")

# print(x_gt_pcs.shape)

dm = DataModule(
    DataModuleConfig(
        cates=[CATE],
        ectconfig=EctConfig(num_thetas=RESOLUTION, bump_steps=RESOLUTION),
        batch_size=16,
    )
)

total = len(dm.test_ds)
idx = 0
x_rendered_pcs = []
x_gt_pcs = []
for test_batch in dm.test_dataloader():
    idx += 1
    x_gt = test_batch.x.view(-1, 2048, 3).type(DTYPE).cuda()

    print(f"Processing idx {idx} out of {total // 16}")
    x_init = (torch.rand(size=(len(x_gt), 2048, 3), dtype=DTYPE) - 0.5).cuda()
    ect_gt = compute_ect_point_cloud(
        x_gt, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE
    )
    # plot_epoch(x_init, x_gt, 0)

    x_rendered = render_point_cloud(
        x_init,
        ect_gt,
        v,
        NUM_EPOCHS,
        scale=SCALE,
        radius=RADIUS,
        resolution=RESOLUTION,
    )

    x_rendered_pcs.append(x_rendered)
    x_gt_pcs.append(x_gt)

x_rendered_pcs = torch.cat(x_rendered_pcs)
x_gt_pcs = torch.cat(x_gt_pcs)
torch.save(
    x_rendered_pcs, f"./results/rendered/reconstructions_{CATE}_{RESOLUTION.item()}.pt"
)
torch.save(x_gt_pcs, f"./results/rendered/references_{CATE}_{RESOLUTION.item()}.pt")

# pl = pv.Plotter(shape=(1, 3), window_size=[1200, 400])
# points = x_rendered_pcs[0].detach().cpu().view(-1, 3).numpy()
# pl.subplot(0, 0)
# actor = pl.add_points(
#     points,
#     style="points",
#     emissive=False,
#     show_scalar_bar=False,
#     render_points_as_spheres=True,
#     scalars=points[:, 2],
#     point_size=5,
#     ambient=0.2,
#     diffuse=0.8,
#     specular=0.8,
#     specular_power=40,
#     smooth_shading=True,
# )
# pl.subplot(0, 1)
# actor = pl.add_points(
#     x_rendered_pcs[0].detach().cpu().view(-1, 3).numpy(),
#     style="points",
#     emissive=False,
#     show_scalar_bar=False,
#     render_points_as_spheres=True,
#     color="lightblue",
#     point_size=5,
#     ambient=0.2,
#     diffuse=0.8,
#     specular=0.8,
#     specular_power=40,
#     smooth_shading=True,
# )
# points = test_batch[0].x.reshape(-1, 3).detach().cpu().numpy()
# actor = pl.add_points(
#     points,
#     style="points",
#     emissive=False,
#     show_scalar_bar=False,
#     render_points_as_spheres=True,
#     color="red",
#     point_size=5,
#     ambient=0.2,
#     diffuse=0.8,
#     specular=0.8,
#     specular_power=40,
#     smooth_shading=True,
# )
# pl.subplot(0, 2)
# actor = pl.add_points(
#     points,
#     style="points",
#     emissive=False,
#     show_scalar_bar=False,
#     render_points_as_spheres=True,
#     scalars=points[:, 2],
#     point_size=5,
#     ambient=0.2,
#     diffuse=0.8,
#     specular=0.8,
#     specular_power=40,
#     smooth_shading=True,
# )

# pl.background_color = "w"
# pl.link_views()
# pl.camera_position = "xy"
# pos = pl.camera.position
# pl.camera.position = (pos[0] + 3, pos[1], pos[2])
# pl.camera.azimuth = 145
# pl.camera.elevation = 20

# # create a top down light
# light = pv.Light(
#     position=(0, 0, 3),
#     positional=True,
#     cone_angle=50,
#     exponent=20,
#     intensity=0.2,
# )
# pl.add_light(light)
# pl.show()


# import imageio

# images = []
# for idx in range(NUM_EPOCHS):
#     if idx % 5 == 0:
#         images.append(imageio.imread(f"./img/{idx}.png"))
# imageio.mimsave("movie.gif", images, fps=5)


# idx = 0
# # data = dm.test_ds[idx]

# recon_pts = torch.load("./results/encoder_chair_sparse/reconstructions.pt")
# ref_pts = torch.load("./results/encoder_chair_sparse/references.pt")

# data_recon = Data(x=recon_pts[idx])
# data_ref = Data(x=ref_pts[idx])

# recon_batch = Batch.from_data_list([data_recon]).cuda()
# ref_batch = Batch.from_data_list([data_ref]).cuda()

# ect_gt = layer(ref_batch, ref_batch.batch, scale=SCALE)


# x = render_point_cloud(
#     ect_gt,
#     layer=layer,
#     num_epochs=NUM_EPOCHS,
#     x_gt=ref_batch[0].x,
#     x_init=recon_batch.x,
#     init_radius=5,
# )
