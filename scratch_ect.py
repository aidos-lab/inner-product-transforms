from layers.ect import EctConfig, EctLayer, compute_ect_point_cloud
from layers.directions import generate_uniform_directions, generate_directions
import matplotlib.pyplot as plt
from torch_geometric.data import Batch, Data


import torch
from torch import nn
import torch.nn.functional as F

# batch_pred = Batch.from_data_list(
#     [Data(x=0.5 * torch.rand(size=(2048, 3), device="cuda:0")) for _ in range(32)]
# ).cuda()

# batch_target = Batch.from_data_list(
#     [Data(x=0.6 * torch.rand(size=(2048, 3), device="cuda:0")) for _ in range(32)]
# ).cuda()


NUM_THETAS = 96
BUMP_STEPS = 96
R = 1.1
v = generate_directions(num_thetas=NUM_THETAS).cuda()

x = torch.rand(size=(3, 2, 3)).cuda()

ect = compute_ect_point_cloud(x, v)

plt.imshow(ect[0].squeeze().detach().cpu().numpy())
plt.show()
