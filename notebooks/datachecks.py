import torch

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# |%%--%%| <wazd8M1wxp|iyUjzvFtYG>


train = torch.load("data/shapenet/airplane/prod/train.pt", weights_only=True)

mean_mean = train.mean(axis=1).mean(axis=0)
mean_std = train.mean(axis=1).std(axis=0)

print("mean mean", mean_mean.tolist())
print("mean std", mean_std.tolist())


# Max radii

scales = (
    (train - train.mean(axis=1, keepdim=True)).norm(dim=-1, keepdim=True).max(axis=1)[0]
).squeeze()

print(scales.shape)

scales_mean = scales.mean()
scales_std = scales.std()

print("scales mean", scales_mean.item())
print("scales std", scales_std.item())
print(scales.max())
print("GlobalRadius", train.norm(dim=-1).max())

# |%%--%%| <iyUjzvFtYG|kyvPfOxWEV>

# def get_datasets(category, dataroot, npoints=2048):
#     tr_dataset = ShapeNet15kPointClouds(
#         root_dir=dataroot,
#         categories=category,
#         split="train",
#         tr_sample_size=npoints,
#         te_sample_size=npoints,
#         scale=1.0,
#         normalize_per_shape=False,
#         normalize_std_per_axis=False,
#         random_subsample=True,
#     )
#     te_dataset = ShapeNet15kPointClouds(
#         root_dir=dataroot,
#         categories=category,
#         split="val",
#         tr_sample_size=npoints,
#         te_sample_size=npoints,
#         scale=1.0,
#         normalize_per_shape=False,
#         normalize_std_per_axis=False,
#         all_points_mean=tr_dataset.all_points_mean,
#         all_points_std=tr_dataset.all_points_std,
#         random_subsample=False,
#     )
#     return te_dataset
#
#
# # dm = load_datamodule(encoder_config.data)
# te_ds = get_datasets(
#     ["airplane"],
#     "../data/shapenet/raw/ShapeNetCore.v2.PC15k",
# )
#
# |%%--%%| <kyvPfOxWEV|xQGnFfamBF>


#
# from layers.directions import generate_uniform_directions
# from layers.ect import compute_ect_point_cloud
# from loaders import load_config
#
# config, _ = load_config("./configs/encoder_airplane.yaml")
#
#
# tr_pc = te_ds[0]["train_points"].view(1, 2048, 3).cuda()
# v = generate_uniform_directions(
#     num_thetas=config.ectconfig.num_thetas,
#     d=config.ectconfig.ambient_dimension,
#     seed=config.ectconfig.seed,
# ).cuda()
# ect = (
#     compute_ect_point_cloud(
#         x=tr_pc,
#         v=v,
#         radius=config.ectconfig.r,
#         resolution=config.ectconfig.resolution,
#         scale=config.ectconfig.scale,
#     )
#     / 2048
# )
#
# # |%%--%%| <xQGnFfamBF|9gGxvFKnSW>
#
# print(config.ectconfig.seed)
# print(config.ectconfig.scale)
#
# # |%%--%%| <9gGxvFKnSW|zepSevym1S>
#
# tr_pc = te_ds[0]["train_points"].view(1, 2048, 3).cuda()
# v = generate_uniform_directions(
#     num_thetas=config.ectconfig.num_thetas,
#     d=config.ectconfig.ambient_dimension,
#     seed=config.ectconfig.seed,
# ).cuda()
# ect = (
#     compute_ect_point_cloud(
#         x=tr_pc,
#         v=v,
#         radius=config.ectconfig.r,
#         resolution=config.ectconfig.resolution,
#         scale=config.ectconfig.scale,
#     )
#     / 2048
# )
#
# # |%%--%%| <zepSevym1S|nwNeHj5FZl>
#
# import matplotlib.pyplot as plt
#
# print(ect.min())
# print(ect.max())
#
# plt.imshow(ect.squeeze().cpu().numpy())
#
#
# # |%%--%%| <nwNeHj5FZl|M5PZ7J99PW>
#
# from loaders import load_datamodule
#
# print(config.data)
# config.data.root = "../data/shapenet"
# dm = load_datamodule(config.data)
#
# # |%%--%%| <M5PZ7J99PW|13LGf3OzQZ>
#
#
# # |%%--%%| <13LGf3OzQZ|k5hxv6sJM7>
#
# pc = dm.val_ds[0].ect
# pts = dm.val_ds[0].x
# # plt.imshow(pc.squeeze().cpu().numpy()-ect.squeeze().cpu().numpy())
#
#
# # |%%--%%| <k5hxv6sJM7|TeI8n8iS4S>
#
#
# # |%%--%%| <TeI8n8iS4S|rKMdiHUMud>
#
# tr_pc = te_ds[0]["test_points"].view(1, 2048, 3).cuda()
# # tr_pc = dm.val_ds[0].x.unsqueeze(0).cuda()
# v = generate_uniform_directions(
#     num_thetas=config.ectconfig.num_thetas,
#     d=config.ectconfig.ambient_dimension,
#     seed=2024,
# ).cuda()
# ect_new = compute_ect_point_cloud(
#     x=tr_pc,
#     v=v,
#     radius=config.ectconfig.r,
#     resolution=config.ectconfig.resolution,
#     scale=64,
# )
#
# ect_new = ect_new / ect_new.max()
#
# print(torch.norm(pc.squeeze().cpu() - ect_new.squeeze().cpu()))
# plt.imshow(pc.squeeze().cpu().numpy() - ect_new.squeeze().cpu().numpy())
#
# # |%%--%%| <rKMdiHUMud|D3shezJ5D1>
#
#
# # |%%--%%| <D3shezJ5D1|RJrvsuTwDR>
#
# plt.imshow(pc.squeeze().cpu().numpy())
#
# # |%%--%%| <RJrvsuTwDR|7P9wSbUsnt>
#
# print(pc.min())
# print(pc.max())
#
#
# # |%%--%%| <7P9wSbUsnt|ryORSDv5Vu>
#
# import pyvista as pv
#
# pv.set_jupyter_backend("server")
# pl = pv.Plotter()
#
# pl.add_points(tr_pc.squeeze().cpu().numpy(), render_points_as_spheres=True, color="red")
# pl.add_points(pts.squeeze().cpu().numpy(), render_points_as_spheres=True)
#
# pl.show()
#
# # |%%--%%| <ryORSDv5Vu|gqS9ngUWO3>
#
# from metrics.evaluation import EMD_CD
#
# old_val = dm.val_ds.x.view(-1, 2048, 3).cuda()
# new_val = torch.cat([data["test_points"].view(1, 2048, 3) for data in te_ds]).cuda()
# print(old_val.shape)
# print(new_val.shape)
# res = EMD_CD(old_val, new_val, batch_size=10, reduced=False)
#
# # |%%--%%| <gqS9ngUWO3|tB8BfWH6qU>
#
# print(res["MMD-CD"].mean())
# print(res["MMD-EMD"].mean())
# d, idxs = torch.topk(res["MMD-CD"], k=10, largest=False)
# print(d)
#
# # |%%--%%| <tB8BfWH6qU|9MpCdJdlKM>
#
# i = 0
#
# import pyvista as pv
#
# pv.set_jupyter_backend("server")
# pl = pv.Plotter()
#
# pl.add_points(
#     old_val[idxs[1]].squeeze().cpu().numpy(), render_points_as_spheres=True, color="red"
# )
# pl.add_points(new_val[idxs[1]].squeeze().cpu().numpy(), render_points_as_spheres=True)
#
# pl.show()
#
# # |%%--%%| <9MpCdJdlKM|MBh9cyr5tW>
