"""
Trains an encoder that takes an ECT and reconstructs a pointcloud for various 
datasets.
"""

import argparse
import json
from types import SimpleNamespace
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
from loaders import (
    load_datamodule,
    load_model,
    load_config,
    load_logger,
    validate_configuration,
)


# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(config: SimpleNamespace, resume=False, debug=False, prod=False):
    """
    Method to train models.
    """
    dm = load_datamodule(config.data)

    if resume:
        model = load_model(
            config.modelconfig,
            f"./{config.trainer.save_dir}/{config.trainer.model_name}",
        ).to(DEVICE)
    else:
        model = load_model(config.modelconfig).to(DEVICE)

    print(model)

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        dirpath="trained_models",
        filename=f"{config.trainer.model_name}"
        + "-epoch={epoch}-val_loss={validation_loss:.2f}",
        auto_insert_metric_name=False,
    )

    if not prod:
        config.loggers.tags.append("dev")

    trainer = L.Trainer(
        logger=None if debug else load_logger(config.loggers),
        callbacks=[checkpoint_callback],
        accelerator=config.trainer.accelerator,
        max_epochs=1 if debug else config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_train_batches=3 if debug else None,
        limit_val_batches=3 if debug else None,
        enable_progress_bar=True,
    )

    trainer.fit(model, dm)
    if not debug:
        trainer.save_checkpoint(
            f"./{config.trainer.save_dir}/{config.trainer.model_name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    parser.add_argument(
        "--prod", default=False, action="store_true", help="Run a test batch"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Run a quick batch"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training"
    )
    args = parser.parse_args()

    run_config, run_config_dict = load_config(args.INPUT)

    train(run_config, resume=args.resume, debug=args.debug, prod=args.prod)
