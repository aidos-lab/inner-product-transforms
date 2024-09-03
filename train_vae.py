"""
Trains a VAE on the ECT's of the mnist dataset.
"""

from dataclasses import dataclass
import argparse
from types import SimpleNamespace
from typing import Any
import json

import yaml
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from models.vae import BaseModel

from datasets import load_datamodule

torch.set_float32_matmul_precision("medium")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_object(dct):
    return SimpleNamespace(**dct)


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


def train(config: Config,resume, dev):
    """
    Method to train variational autoencoders.
    """

    dm = load_datamodule(config.data)

    if resume:
        litmodel = BaseModel.load_from_checkpoint(f"./{config.trainer.save_dir}/{config.trainer.save_name}.ckpt")
    else:
        litmodel = BaseModel(
        config.vaeconfig, config.ectconfig, config.trainer.max_epochs, config.trainer.learning_rate
    )

    # Set up debug percentages
    limit_train_batches = None

    if dev:
        limit_train_batches = 0.1 
        config.trainer.max_epochs = 1


    trainer = L.Trainer(
        logger=TensorBoardLogger("my_logs", name=f"{config.trainer.experimentname}"),
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_train_batches=limit_train_batches,
        fast_dev_run=False,
    )

    trainer.fit(litmodel, dm)
    trainer.save_checkpoint(f"./{config.trainer.save_dir}/{config.trainer.save_name}.ckpt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a test batch"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training"
    )
    args = parser.parse_args()


    with open(args.INPUT, encoding="utf-8") as stream:
        run_dict = yaml.safe_load(stream)
        run_config = json.loads(json.dumps(run_dict), object_hook=load_object)

    train(run_config,resume=args.resume, dev=args.dev)

