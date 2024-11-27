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
<<<<<<< HEAD
from models.encoder_sparse import BaseModel
=======
from models.encoder import BaseModel

# from load_model_scaled import load_encoder
>>>>>>> 07a15a404b153199d5916bd22c7b6446209d0ae6


# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



def load_object(dct):
    return SimpleNamespace(**dct)


def train(config: SimpleNamespace, resume=False, dev=False, path=""):
    """
    Method to train variational autoencoders.
    """
    dm = load_datamodule(config.data)

    if resume:
        # model = load_encoder(path)
        model = BaseModel.load_from_checkpoint(
            f"./{config.trainer.save_dir}/{config.trainer.save_name}.ckpt"
        ).to(DEVICE)
    else:
        model = BaseModel(
            ectconfig=config.ectconfig,
            ectlossconfig=config.ectlossconfig,
            modelconfig=config.modelconfig,
        )

    # Set up debug percentages
    limit_train_batches = None

    if dev:
        limit_train_batches = 0.1
        config.trainer.max_epochs = 1

    trainer = L.Trainer(
        logger=TensorBoardLogger(
            "my_logs", name=f"{config.trainer.experimentname}"
        ),
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0.1,
    )

    print(config)

    trainer.fit(model, dm)
    trainer.save_checkpoint(
        f"./{config.trainer.save_dir}/{config.trainer.save_name}.ckpt"
    )


<<<<<<< HEAD
def main():
=======
if __name__ == "__main__":
>>>>>>> 07a15a404b153199d5916bd22c7b6446209d0ae6
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

    train(run_config, resume=args.resume, dev=args.dev, path=args.INPUT)
<<<<<<< HEAD


if __name__ == "__main__":
    main()
=======
>>>>>>> 07a15a404b153199d5916bd22c7b6446209d0ae6
