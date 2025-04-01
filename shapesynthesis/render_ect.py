import torch, time, gc
from layers.directions import generate_uniform_directions
from layers.ect import compute_ect_point_cloud
from loaders import load_config
from torch import nn

from plotting import plot_recon_3d

NUM_EPOCHS = 200
NUM_PTS = 2048
USE_AMP = True

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


# # Timing utilities
# start_time = None


# def start_timer():
#     global start_time
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     torch.cuda.synchronize()
#     start_time = time.time()


# def end_timer_and_print(local_msg):
#     torch.cuda.synchronize()
#     end_time = time.time()
#     print("\n" + local_msg)
#     print("Total execution time = {:.3f} sec".format(end_time - start_time))
#     print(
#         "Max memory used by tensors = {} bytes".format(
#             torch.cuda.max_memory_allocated()
#         )
#     )


def render_point_cloud_viz(x_init, ect_gt, v, num_epochs, scale, radius, resolution):
    # x = [nn.Parameter(x_in.unsqueeze(0)) for x_in in x_init]
    result = [x_init.clone()]
    x = nn.Parameter(x_init)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=[x],
        lr=0.1,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1000], gamma=0.5
    )

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        ect_pred = compute_ect_point_cloud(
            x, v, radius=radius, resolution=resolution, scale=scale
        )
        loss = loss_fn(ect_pred, ect_gt)
        loss.backward()
        optimizer.step()
        result.append(x.detach().clone())

        scheduler.step()
    return x.detach(), torch.cat(result, dim=0)


# ############################################################################ #
#                                   Data prep                                  #
# ############################################################################ #

config, _ = load_config("./shapesynthesis/configs/encoder_airplane.yaml")
ectconfig = config.data.ectconfig
ectconfig.scale = 64
ectconfig.r = 1.1


v = generate_uniform_directions(
    num_thetas=ectconfig.num_thetas,
    d=ectconfig.ambient_dimension,
    seed=ectconfig.seed,
).cuda()


# Load the ECT's
ref_pc = torch.load("./results/encoder_airplane/references.pt").cuda()[0]
print(ref_pc.shape)
print(ref_pc.mean(dim=0))
print(ref_pc.norm(dim=-1).max())

ref_pc -= ref_pc.mean(dim=0)
ref_pc /= ref_pc.norm(dim=-1).max()

ref_pc = ref_pc.unsqueeze(0)

print(ref_pc.shape)

ect_gt = compute_ect_point_cloud(ref_pc, v, ectconfig.r, 256, scale=ectconfig.scale)

print(ect_gt.shape)
print("ECT min", ect_gt.min())
print("ECT max", ect_gt.max())

x_init = (
    torch.rand(size=(len(ect_gt), NUM_PTS, ectconfig.ambient_dimension)) - 0.5
).cuda()


x, vis = render_point_cloud_viz(
    x_init=x_init,
    ect_gt=ect_gt,
    v=v,
    scale=ectconfig.scale,
    num_epochs=NUM_EPOCHS,
    radius=ectconfig.r,
    resolution=256,
)

torch.save(vis, "./results/rendered_ect/render.pt")

plot_recon_3d(
    vis.detach().cpu().numpy(), vis.detach().cpu().numpy(), num_pc=9, offset=90
)


# ############################################################################ #
#                                   Wildlands                                  #
# ############################################################################ #

# config, _ = load_config("./shapesynthesis/configs/vae_airplane.yaml")
# ectconfig = config.data.ectconfig
# ectconfig.scale = 64
# ectconfig.r = 1.1
# print(ectconfig)

# v = generate_uniform_directions(
#     num_thetas=ectconfig.num_thetas,
#     d=ectconfig.ambient_dimension,
#     seed=ectconfig.seed,
# ).cuda()


# # Load the ECT's
# ref_pc = torch.load("./results/encoder_airplane/references.pt").cuda()[:9]

# # scale
# center = torch.mean(ref_pc, dim=1, keepdim=True)
# ref_pc -= center
# print(ref_pc.shape)
# scale = torch.norm(ref_pc, dim=-1, keepdim=True).max(dim=1)[0]
# print(scale.shape)
# ref_pc /= scale.unsqueeze(1)

# print(ref_pc.shape)
# ects = compute_ect_point_cloud(ref_pc, v, ectconfig.r, 256, scale=ectconfig.scale) / 2


# print(ects.shape)

# print("ECT min", ects.min())
# print("ECT max", ects.max())
# # print(len(ects))
# # # plt.imshow(ects[0].squeeze().cpu().numpy())
# # # plt.show()

# x_init = (
#     torch.rand(size=(len(ects), NUM_PTS, ectconfig.ambient_dimension)) - 0.5
# ).cuda()

# (x_rendered, x_render), elapsed_time = timed(
#     lambda: render_point_cloud_viz(
#         x_init,
#         ects,
#         v,
#         NUM_EPOCHS,
#         scale=ectconfig.scale,
#         radius=1.1,
#         resolution=ectconfig.resolution,
#     )
# )

# print("Elapsed:", elapsed_time)

# torch.save(x_render, "render.pt")


# @torch.compile()
# def render_point_cloud(
#     x_init, ect_gt, v, num_epochs, scale, radius, resolution, use_amp=False
# ):
#     loss_fn = torch.nn.MSELoss()
#     x = nn.Parameter(x_init)
#     optimizer = torch.optim.Adam(
#         params=[x],
#         lr=0.00005,
#     )

#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=[50, 100, 200, 1000], gamma=0.5
#     )
#     scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

#     for epoch in range(1, num_epochs + 1):
#         optimizer.zero_grad()
#         with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
#             ect_pred = compute_ect_point_cloud(
#                 x,
#                 v,
#                 radius=radius,
#                 resolution=resolution,
#                 scale=scale,
#             )
#             # print(ect_pred.max())
#             loss = loss_fn(ect_pred, ect_gt)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         old_scaler = scaler.get_scale()
#         scaler.update()
#         new_scaler = scaler.get_scale()
#         if old_scaler == new_scaler:
#             scheduler.step()
#         # if epoch % 5 == 0:
#         #     result.append(x.detach().clone())
#     return x.detach()
