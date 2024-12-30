import matplotlib.pyplot as plt
import torch
from datasets.mnist import DataModule, DataModuleConfig
from layers.directions import generate_uniform_directions
from layers.ect import EctConfig, compute_ect_point_cloud
from plotting import plot_recon_2d
from renderers.pointcloud import render_point_cloud

NUM_EPOCHS = 1000
RESOLUTION = 32
SCALE = 16
IDX = 8
SEED = 2024
RADIUS = 1.1
DTYPE = torch.float32
NUM_PTS = 32
DIM = 2

v = (
    generate_uniform_directions(num_thetas=RESOLUTION, d=DIM, seed=SEED)
    .type(DTYPE)
    .cuda()
)
# x_gt_pcs = torch.load("./results/encoder_chair_sparse/references.pt")


# print(x_gt_pcs.shape)
dm = DataModule(
    DataModuleConfig(
        module="datasets.mnist",
        num_pts=128,
        pin_memory=True,
        drop_last=False,
        force_reload=False,
        num_workers=0,
        root="./data/mnistpointcloud",
        ectconfig=EctConfig(
            num_thetas=32,
            r=1.1,
            scale=SCALE,
            ect_type="points",
            resolution=RESOLUTION,
            ambient_dimension=2,
            normalized=True,
            seed=SEED,
        ),
        batch_size=16,
    ),
    force_reload=False,
)

total = len(dm.test_ds)


idx = 0
x_rendered_pcs = []
x_gt_pcs = []
for test_batch in dm.test_dataloader():
    idx += 1
    x_gt = test_batch.x.view(-1, NUM_PTS, DIM).type(DTYPE).cuda()

    print(f"Processing idx {idx} out of {total // 16}")
    x_init = (
        torch.rand(size=(len(x_gt), NUM_PTS, DIM), dtype=DTYPE) - 0.5
    ).cuda()
    ect_gt = compute_ect_point_cloud(
        x_gt, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE
    )

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
    break

x_rendered_pcs = torch.cat(x_rendered_pcs)
x_gt_pcs = torch.cat(x_gt_pcs)


print(ect_gt.shape)

fig = plot_recon_2d(x_rendered_pcs.cpu().detach(), x_gt_pcs.cpu().detach())
plt.show()

# torch.save(x_rendered_pcs, f"./results/rendered_mnist/reconstructions_{RESOLUTION}.pt")
# torch.save(x_gt_pcs, f"./results/rendered_mnist/references_{RESOLUTION}.pt")
