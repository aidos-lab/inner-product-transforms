
import argparse
from dataclasses import dataclass
from typing import Any
import torch
import lightning as L

from omegaconf import OmegaConf

from datasets import load_datamodule
from models.encoder import BaseModel
from layers.ect import EctLayer, EctConfig

from layers.directions import generate_directions
from loggers import get_wandb_logger

from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
import os
import torch
import numpy as np
import torch.nn as nn


# Settings
torch.set_float32_matmul_precision("medium")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test(config):
    """
    Method to train variational autoencoders.
    """
    dm = load_datamodule(config.data)

    layer = EctLayer(
        EctConfig(
            num_thetas=config.layer.ect_size,
            bump_steps=config.layer.ect_size,
            normalized=True,
            device=DEVICE,
        ),
        v=generate_directions(config.layer.ect_size, config.layer.dim, DEVICE),
    )
    

    model = BaseModel.load_from_checkpoint(
        f"./trained_models/ectencoder_shapenet_{config.data.categories[0]}.ckpt",
        layer=layer,
        ect_size=config.layer.ect_size,
        hidden_size=config.model.hidden_size,
        num_pts=config.model.num_pts,
        num_dims=config.model.num_dims,
        learning_rate=config.model.learning_rate,
    ).to(DEVICE)
    model.cuda()
    model.eval()
   
    all_sample = []
    all_ref = []


    for batch in dm.val_dataloader():
        batch = batch.cuda()
        ect = layer(batch, batch.batch)
        encoder_pointcloud = model.model(ect).view(-1, 2048, 3)   
        print(batch.x.view(-1,2048,3).norm(dim=-1).max(dim=-1)[0])

        all_sample.append(encoder_pointcloud)
        all_ref.append(batch.x.view(-1,2048,3))

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    results = EMD_CD(
            sample_pcs, ref_pcs, 64, reduced=True, accelerated_cd=True
        )

    print(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    args = parser.parse_args()
    config = OmegaConf.load(args.INPUT)
    test(config)
