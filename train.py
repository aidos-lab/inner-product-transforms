"""
Trains an encoder that takes an ECT and reconstructs a pointcloud for various 
datasets.
"""

import argparse
from types import SimpleNamespace
import torch
import lightning as L
from loaders import load_datamodule, load_model, load_config, load_logger


# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(config: SimpleNamespace, resume=False, dev=False):
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

    logger = load_logger(config.loggers, logger_type="tensorboard")

    limit_train_batches = None

    if dev:
        limit_train_batches = 0.1
        config.trainer.max_epochs = 1

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0.1,
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(f"./{config.trainer.save_dir}/{config.trainer.model_name}")


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

    run_config = load_config(args.INPUT)
    train(run_config, resume=args.resume, dev=args.dev)
