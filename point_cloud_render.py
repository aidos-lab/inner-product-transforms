import torch
from layers.directions import generate_uniform_directions
import torch
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud
from datasets.shapenetcore import DataModule, DataModuleConfig

RESOLUTION = 256
NUM_EPOCHS = 600
SCALE = 128
RADIUS = 4.5
CATE = "car"
v = generate_uniform_directions(num_thetas=256, d=3, seed=20).cuda()

dm = DataModule(DataModuleConfig(cates=[CATE]))
x_gt_pcs = dm.test_ds.x.view(-1, 2048, 3).cuda()

means = torch.tensor(dm.test_ds.mean).cuda()
stdev = torch.tensor(dm.test_ds.std).cuda()

torch.save(means, f"./results/rendered_{CATE}/means.pt")
torch.save(stdev, f"./results/rendered_{CATE}/stdevs.pt")

x_rendered_pcs = []
for idx, x_gt in enumerate(x_gt_pcs):
    print(f"Processing idx {idx}")
    x_init = RADIUS * (torch.rand(size=(1, 2048, 3)) - 0.5).cuda()
    ect_gt = compute_ect_point_cloud(
        x_gt.view(1, 2048, 3), v, radius=RADIUS, resolution=256, scale=SCALE
    )
    num_pts = 2048
    x_rendered = render_point_cloud(
        x_init, ect_gt, v, num_pts, NUM_EPOCHS, x_gt, scale=SCALE, radius=RADIUS
    )
    x_rendered_pcs.append(x_rendered)


x_rendered_pcs = torch.cat(x_rendered_pcs)
torch.save(x_rendered_pcs, f"./results/rendered_{CATE}/reconstructions.pt")
torch.save(x_gt_pcs, f"./results/rendered_{CATE}/references.pt")
