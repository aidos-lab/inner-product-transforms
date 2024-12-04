import torch
from layers.directions import generate_uniform_directions
import pyvista as pv
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud
from datasets.shapenetcore import DataModule, DataModuleConfig
from layers.ect import EctConfig

RESOLUTION = 256
NUM_EPOCHS = 100
SCALE = 128
IDX = 8

v = generate_uniform_directions(num_thetas=256, d=3, seed=20).cuda()
# x_gt_pcs = torch.load("./results/encoder_chair_sparse/references.pt")

# print(x_gt_pcs.shape)

dm = DataModule(
    DataModuleConfig(
        cates=["airplane"],
        ectconfig=EctConfig(num_thetas=128, bump_steps=128),
    )
)

for test_batch in dm.test_dataloader():
    break

x_gt_pcs = test_batch[0].x.view(1, 2048, 3).cuda()

x_rendered_pcs = []
for idx, x_gt in enumerate(x_gt_pcs):
    print(f"Processing idx {idx}")
    x_init = 5 * (torch.rand(size=(1, 2048, 3)) - 0.5).cuda()
    ect_gt = compute_ect_point_cloud(
        x_gt.view(1, 2048, 3), v, radius=5, resolution=256, scale=SCALE
    )
    num_pts = 2048
    x_rendered = render_point_cloud(
        x_init, ect_gt, v, num_pts, NUM_EPOCHS, x_gt, scale=SCALE
    )
    x_rendered_pcs.append(x_rendered)


# x_rendered_pcs = torch.cat(x_rendered_pcs)
# torch.save(x_rendered_pcs, "./results/rendered_chair/reconstructions.pt")
# torch.save(x_gt_pcs, "./results/rendered_chair/references.pt")

pl = pv.Plotter(shape=(1, 3), window_size=[1200, 400])
points = x_rendered_pcs[0].detach().cpu().view(-1, 3).numpy()
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
    x_rendered_pcs[0].detach().cpu().view(-1, 3).numpy(),
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
points = test_batch[0].x.reshape(-1, 3).detach().cpu().numpy()
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
    position=(0, 0, 3),
    positional=True,
    cone_angle=50,
    exponent=20,
    intensity=0.2,
)
pl.add_light(light)
pl.show()


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
