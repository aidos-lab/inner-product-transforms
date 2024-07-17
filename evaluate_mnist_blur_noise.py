"""

- import dataset, model and base stuff
- per class and 10 steps compute 
    added noise and chamfer distance    
- output the result in a table and csv

inputs:
- Type of noise
- Config (data & model)

"""

from omegaconf import OmegaConf
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Batch, Data
from torchvision.transforms import GaussianBlur

from kaolin.metrics.pointcloud import chamfer_distance

from torch_scatter import scatter_mean

from datasets.mnist import DataModule, DataModuleConfig
from models.encoder import BaseModel as EctEncoder
from layers.ect import EctLayer, EctConfig
from layers.directions import generate_directions

DEVICE = "cuda:0"

# Set to -1 to evaluate on the whole dataset.
NUM_TEST_SAMPLES = 100

dm = DataModule(DataModuleConfig(root="./data/mnistpointcloud"))

encoder_config = OmegaConf.load("./configs/config_encoder_mnist.yaml")

layer = EctLayer(
    EctConfig(
        num_thetas=encoder_config.layer.ect_size,
        bump_steps=encoder_config.layer.ect_size,
        normalized=True,
        device=DEVICE,
    ),
    v=generate_directions(
        encoder_config.layer.ect_size, encoder_config.layer.dim, DEVICE
    ),
)

## Load the encoder

ect_encoder_litmodel = EctEncoder.load_from_checkpoint(
    f"./trained_models/{encoder_config.model.save_name}",
    layer=layer,
    ect_size=encoder_config.layer.ect_size,
    hidden_size=encoder_config.model.hidden_size,
    num_pts=encoder_config.model.num_pts,
    num_dims=encoder_config.model.num_dims,
    learning_rate=encoder_config.model.learning_rate,
).to(DEVICE)


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


class GaussianTransform:
    def __init__(self, k=1):
        self.blur = GaussianBlur(kernel_size=5, sigma=2)
        self.layer = layer
        self.k = k

    def __call__(self, data):
        ect = self.layer(data, torch.zeros(size=(data.x.shape[0],), dtype=torch.int))
        ect = ect + 0.05 * torch.randn_like(ect)
        new_data = Data(x=data.x, ect=ect)
        return ect, None


################################################################################
#### Script starts here.
################################################################################


data_list = [data for data in dm.test_ds]
y = dm.test_ds[:].y

y = y[:NUM_TEST_SAMPLES]
data_list = data_list[:NUM_TEST_SAMPLES]

# Visualizatioon
idxs = torch.hstack(
    [
        torch.where(y == 0)[0][:1],
        torch.where(y == 1)[0][:1],
        torch.where(y == 2)[0][:1],
        torch.where(y == 3)[0][:1],
        torch.where(y == 4)[0][:1],
        torch.where(y == 5)[0][:1],
        torch.where(y == 6)[0][:1],
        torch.where(y == 7)[0][:1],
        torch.where(y == 8)[0][:1],
        torch.where(y == 9)[0][:1],
    ]
)


vis_x = {i: [] for i in range(10)}
vis_ect = {i: [] for i in range(10)}

batch = Batch.from_data_list(data_list=data_list).cuda()

ch_list = []


blur = GaussianBlur(kernel_size=5, sigma=2)

base_ect = layer(batch, batch.batch)
noised_ect = base_ect

for _ in range(10):
    noised_ect = blur(noised_ect)

    with torch.no_grad():
        recon_batch = ect_encoder_litmodel.model.forward(noised_ect).cuda()

    for i in vis_x.keys():
        vis_x[i].append(recon_batch[idxs[i]])
        vis_ect[i].append(noised_ect[idxs[i]])

    loss_ch = chamfer_distance(
        torch.cat(
            [
                recon_batch.view(len(recon_batch), 128, 2),
                torch.zeros(size=(len(recon_batch), 128, 1), device="cuda"),
            ],
            dim=-1,
        ),
        torch.cat(
            [
                batch.x.view(-1, 128, 2),
                torch.zeros(size=(len(batch), 128, 1), device="cuda"),
            ],
            dim=-1,
        ),
    )

    ch_list.append(scatter_mean(loss_ch.cpu(), y))


ch_list = [el.detach().numpy() for el in ch_list]

columns = [
    "DigitZero",
    "DigitOne",
    "DigitTwo",
    "DigitThree",
    "DigitFour",
    "DigitFive",
    "DigitSix",
    "DigitSeven",
    "DigitEight",
    "DigitNine",
]

df = pd.DataFrame(np.vstack(ch_list), columns=columns)
df1 = pd.concat([df, df.apply(["mean", "std"], axis=0)])
df2 = pd.concat([df1, df1.apply(["mean", "std"], axis=1)], axis=1)

df2.to_csv(
    f"./results/noise_blur.csv", float_format="%.4f", index=True, index_label="Index"
)

torch.save(vis_x, f"./results/noise_blur_vis_x.pt")
torch.save(vis_ect, f"./results/noise_blur_vis_ect.pt")
