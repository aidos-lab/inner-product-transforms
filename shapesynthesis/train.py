"""
Trains an encoder that takes an ECT and reconstructs a pointcloud for various
datasets.
"""

import argparse
import os
from types import SimpleNamespace

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from loaders import (
    load_config,
    load_datamodule,
    load_logger,
    load_model,
    validate_configuration,
)

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(config: SimpleNamespace, resume=False, dev=False, prod=False):
    """
    Method to train models.
    """

    # Modify configs based on dev flag.
    if dev:
        config.loggers.tags.append("dev")
        config.trainer.max_epochs = 1
        config.trainer.save_dir += "_dev"

    # Create trained models directory if it does not exist yet.
    os.makedirs(config.trainer.save_dir, exist_ok=True)

    dm = load_datamodule(config.data, dev=dev)

    if resume:
        model = load_model(
            config.modelconfig,
            f"./{config.trainer.save_dir}/{config.trainer.model_name}",
        ).to(DEVICE)
    else:
        model = load_model(config.modelconfig).to(DEVICE)

    logger = load_logger(config.loggers)

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    trainer.fit(model, dm)

    print("SAVING TO:", f"./{config.trainer.save_dir}/{config.trainer.model_name}")
    trainer.save_checkpoint(f"./{config.trainer.save_dir}/{config.trainer.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training"
    )
    args = parser.parse_args()
    run_config, run_config_dict = load_config(args.INPUT)
    train(run_config, resume=args.resume, dev=args.dev)
