"""
Trains an encoder that takes an ECT and reconstructs a pointcloud for various
datasets.
"""

import argparse
import os
from types import SimpleNamespace

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from loaders import load_config, load_datamodule, load_logger, load_model

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(config: SimpleNamespace, resume=False, evaluate=False, dev=False):
    """
    Method to train models.
    """

    # Modify configs based on dev flag.
    if dev:
        config.loggers.tags.append("dev")
        # config.trainer.max_epochs = 1
        config.trainer.save_dir += "_dev"
        config.trainer.results_dir += "_dev"

    result_dir = f"{config.trainer.results_dir}/{config.loggers.experiment_name}"
    os.makedirs(result_dir, exist_ok=True)

    print(80 * "=")
    print("Config used (with mods):")
    print(80 * "=")
    print(config.modelconfig)
    print(80 * "=")

    dm = load_datamodule(config.data, dev=dev)

    if resume:
        model = load_model(
            config.modelconfig,
            f"./{config.trainer.save_dir}/{config.trainer.model_name}",
        ).to(DEVICE)
    else:
        model = load_model(config.modelconfig).to(DEVICE)

    logger = load_logger(config.loggers)

    class SaveTestOutput(L.Callback):
        def __init__(self, results_dir: str):
            super().__init__()
            self.results_dir = results_dir
            self.ground_truth_batches = []
            self.predicted_batches = []

        def on_test_batch_end(
            self,
            trainer,
            pl_module,
            outputs: tuple,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            self.ground_truth_batches.append(outputs[0])
            self.predicted_batches.append(outputs[1])
            return super().on_test_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

        def on_test_end(self, trainer, pl_module):
            print("RUN EPOCH END")
            gt = torch.cat(self.ground_truth_batches)
            pred = torch.cat(self.predicted_batches)
            torch.save(gt, f"{self.results_dir}/ground_truth.pt")
            torch.save(pred, f"{self.results_dir}/predictions.pt")
            return super().on_test_end(trainer, pl_module)

    trainer = L.Trainer(
        logger=logger,
        callbacks=[
            # SaveTestOutput(results_dir=result_dir),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        check_val_every_n_epoch=5,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    trainer.fit(
        model, train_dataloaders=dm.train_dataloader, val_dataloaders=dm.val_dataloader
    )

    # Set up for testing
    trainer.test(model, dataloaders=dm.test_dataloader)

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
    parser.add_argument(
        "--evaluate",
        default=False,
        action="store_true",
        help="Evaluate the model after training.",
    )
    args = parser.parse_args()
    run_config, _ = load_config(args.INPUT)

    train(run_config, resume=args.resume, dev=args.dev, evaluate=args.evaluate)
