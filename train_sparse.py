"""
Trains an encoder that takes an ECT and reconstructs a pointcloud for various 
datasets.
"""

import argparse
from dataclasses import dataclass
from typing import Any
import torch
import lightning as L

from omegaconf import OmegaConf

from datasets import load_datamodule
from models.encoder_sparse import BaseModel
from layers.ect import EctLayer, EctConfig

from layers.directions import generate_directions
from loggers import get_wandb_logger


# Settings
torch.set_float32_matmul_precision("medium")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """
    Interface for the configurations in the yaml file.
    """

    layer: Any
    data: Any
    model: Any
    loggers: Any
    trainer: Any


# @dataclass
# class ModelConfig:
#     """
#     Interface for the configurations in the yaml file.
#     """

#     layer: Any
#     data: Any
#     model: Any
#     litmodel: Any
#     loggers: Any
#     trainer: Any

# @dataclass
# class DataConfig:
#     """
#     Interface for the configurations in the yaml file.
#     """

#     layer: Any
#     data: Any
#     model: Any
#     litmodel: Any
#     loggers: Any
#     trainer: Any


# @dataclass
# class ModelConfig:
#     """
#     Interface for the configurations in the yaml file.
#     """

#     layer: Any
#     data: Any
#     model: Any
#     litmodel: Any
#     loggers: Any
#     trainer: Any


def train(config: Config):
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

    model = BaseModel(
        layer=layer,
        ect_size=config.layer.ect_size,
        hidden_size=config.model.hidden_size,
        num_pts=config.model.num_pts,
        num_dims=config.model.num_dims,
        learning_rate=config.model.learning_rate,
    )

    logger = get_wandb_logger(config.loggers)

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        fast_dev_run=False,
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(f"./trained_models/{config.model.save_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    args = parser.parse_args()
    config = OmegaConf.load(args.INPUT)
    train(config)
