"""
Trains a VAE on the ECT's of the mnist dataset.
"""

from dataclasses import dataclass
import argparse
from typing import Any
from omegaconf import OmegaConf
import torch
import yaml
import lightning as L
from layers.ect import EctLayer, EctConfig
from layers.directions import generate_directions
from loggers import get_wandb_logger
from models.vae import VanillaVAE, BaseModel
from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_kld_loss_fn

from datasets import load_datamodule

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
    litmodel: Any
    loggers: Any
    trainer: Any


def train(config: Config):
    """
    Method to train variational autoencoders.
    """

    layer = EctLayer(
        EctConfig(
            num_thetas=config.layer.ect_size,
            bump_steps=config.layer.ect_size,
            normalized=True,
            device=DEVICE,
        ),
        v=generate_directions(config.layer.ect_size, config.layer.dim, DEVICE),
    )

    dm = load_datamodule(config.data)

    model = VanillaVAE(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        img_size=config.layer.ect_size,
    )

    litmodel = BaseModel(
        model,
        *get_mse_metrics(),
        accuracies_fn=compute_mse_accuracies,
        loss_fn=compute_mse_kld_loss_fn,
        learning_rate=config.litmodel.learning_rate,
        layer=layer,
    )

    logger = get_wandb_logger(config.loggers)

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        fast_dev_run=False,
    )

    trainer.fit(litmodel, dm)
    trainer.save_checkpoint(f"./trained_models/{config.model.save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    args = parser.parse_args()
    config = OmegaConf.load(args.INPUT)
    train(config)
