"""
Trains an encoder that takes an ECT and reconstructs a pointcloud for various 
datasets.
"""

import json
import argparse
from types import SimpleNamespace
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import lightning as L
from datasets import load_datamodule
from models.encoder import BaseModel

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_object(dct):
    return SimpleNamespace(**dct)

def train(config: SimpleNamespace, resume, dev):
    """
    Method to train variational autoencoders.
    """
    dm = load_datamodule(config.data)

    if resume:
        model = BaseModel.load_from_checkpoint(f"./{config.trainer.save_dir}/{config.trainer.save_name}.ckpt").to(DEVICE)
    else:
        model = BaseModel(
            ectconfig=config.ectconfig,
            ectlossconfig=config.ectlossconfig,
            modelconfig=config.modelconfig,
        )

    # logger = get_wandb_logger(config.loggers)

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
    )

    trainer.fit(model, dm)
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
