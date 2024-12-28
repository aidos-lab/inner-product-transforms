import torch
from layers.directions import generate_uniform_directions
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud

# torch.set_float32_matmul_precision("high")
NUM_EPOCHS = 2000
RESOLUTION = 128
SCALE = int(RESOLUTION * 0.25)
IDX = 8
SEED = 2013
RADIUS = torch.tensor(7)
DTYPE = torch.float16
CATE = "car"
RADIUS = 7
BATCH_SIZE = 32
v = (
    generate_uniform_directions(num_thetas=RESOLUTION, d=3, seed=SEED)
    .type(DTYPE)
    .cuda()
)

# dm = DataModule(
#     DataModuleConfig(
#         cates=[CATE],
#         ectconfig=EctConfig(
#             num_thetas=RESOLUTION,
#             resolution=RESOLUTION,
#             r=7,
#             scale=SCALE,
#             ect_type="points",
#             ambient_dimension=3,
#             normalized=True,
#             seed=SEED,
#         ),
#         batch_size=BATCH_SIZE,
#     )
# )

backend_kwargs = {
    # "enabled_precisions": {torch.half},
    "debug": True,
    # "min_block_size": 2,
    # "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
    # "optimization_level": 4,
    # "use_python_runtime": False,
}

compute_ect_point_cloud_compiled = torch.compile(
    compute_ect_point_cloud,
    backend="inductor",
    # options=backend_kwargs,
    # dynamic=False,
)

total = 1
idx = 0
x_rendered_pcs = []
x_gt_pcs = []
# for test_batch in dm.test_dataloader():
#     break

test_batch = torch.load("batch.pt")

idx += 1
x_gt = test_batch.x.view(-1, 2048, 3).type(DTYPE).cuda()
x_gt = torch.cat([x_gt, x_gt])

print(x_gt.shape)


print(f"Processing idx {idx} out of {total // 16}")
x_init = (torch.rand(size=(len(x_gt), 2048, 3), dtype=DTYPE) - 0.5).cuda()
ect_gt = compute_ect_point_cloud_compiled(
    x_gt,
    v,
    radius=RADIUS,
    resolution=RESOLUTION,
    scale=torch.tensor(SCALE).cuda(),
)


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


for _ in range(5):
    x_rendered, elapsed = timed(
        lambda: render_point_cloud(
            x_init,
            ect_gt,
            v,
            NUM_EPOCHS,
            scale=SCALE,
            radius=RADIUS,
            resolution=RESOLUTION,
        )
    )
    print(x_rendered.dtype)

    print(elapsed)

x_rendered_pcs.append(x_rendered)
x_gt_pcs.append(x_gt)

x_rendered_pcs = torch.cat(x_rendered_pcs)
x_gt_pcs = torch.cat(x_gt_pcs)

# torch.save(
#     x_rendered_pcs,
#     f"./results/rendered/reconstructions_{CATE}_{RESOLUTION.item()}.pt",
# )
# torch.save(
#     x_gt_pcs, f"./results/rendered/references_{CATE}_{RESOLUTION.item()}.pt"
# )

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
