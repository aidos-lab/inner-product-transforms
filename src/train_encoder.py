"""
Trains the encoder given a configuration from the config folder.
For training your own model, have a look at the encoder_airplane
configuration.

"""

import argparse
import os

import lightning
import pydantic
import torch
import torch.nn.functional as F
import torchvision
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid
from tqdm import tqdm

from loaders import (
    load_config,
    load_datamodule,
    load_logger,
    load_model,
    load_transform,
)
from metrics.loss import chamfer
from plotting import plot_recon_3d
from transforms.ecttransform import Transform, TransformConfig

torch.set_float32_matmul_precision("medium")


def train(
    fabric,
    dataloader,
    transform,
    model,
    optimizer,
    losstransform,
    ecttransform,
    trainerconfig,
    loggerconfig,
    logger,
    no_progressbar,
    results_base_dir,
    dev,
):

    step = 0
    for epoch in range(trainerconfig.max_epochs):
        batch_idx = -1
        for pcs in tqdm(dataloader, disable=no_progressbar):
            optimizer.zero_grad(set_to_none=True)
            batch_idx += 1
            pcs = transform(pcs)

            ect = ecttransform(pcs).unsqueeze(1)
            pcs_recon = model(ect)

            ect_gt = losstransform(pcs)
            ect_recon = losstransform(pcs_recon)

            # Loss terms
            ect_loss = F.mse_loss(ect_gt, ect_recon)
            cd_loss = chamfer(pcs_recon, pcs)

            loss = cd_loss + 10 * ect_loss

            fabric.backward(loss)
            optimizer.step()

            logger.log_metrics(
                {
                    f"loss": loss,
                    f"ect_loss": ect_loss,
                    f"cd_loss": cd_loss,
                },
                step=step,
            )
            step += 1

            # Save model.
            if batch_idx == 0 and epoch % trainerconfig.checkpoint_interval == 0:
                state = {"model": model}
                if not dev:
                    print(
                        f" Saving model to {results_base_dir}/model.ckpt",
                    )
                    fabric.save(
                        f"{results_base_dir}/model.ckpt",
                        state,
                    )
                print(
                    f"Saving screenshot to: {results_base_dir}/pcs_recon_{epoch:04}.png"
                )
                plot_recon_3d(
                    pcs_recon[:8],
                    pcs[:8],
                    num_pc=8,
                    filename=f"{results_base_dir}/pcs_recon_{epoch:04}.png",
                )
    # Save model.
    state = {"model": model}
    fabric.save(
        f"{results_base_dir}/model.ckpt",
        state,
    )


@torch.no_grad()
def test(
    fabric,
    dataloader,
    transform,
    model,
    optimizer,
    losstransform,
    ecttransform,
    trainerconfig,
    loggerconfig,
    logger,
    no_progressbar,
    results_base_dir,
    dev,
):

    model.eval()
    batch_idx = -1
    pcs_recon_list = []
    pcs_gt_list = []
    for pcs in tqdm(dataloader, disable=no_progressbar):
        batch_idx += 1

        ect = ecttransform(pcs).unsqueeze(1)
        pcs_recon = model(ect)

        ect_gt = losstransform(pcs)
        ect_recon = losstransform(pcs_recon)

        # Loss terms
        ect_loss = F.mse_loss(ect_gt, ect_recon)
        cd_loss = chamfer(pcs_recon, pcs)

        loss = cd_loss + 10 * ect_loss

        pcs_recon_list.append(pcs_recon.detach().cpu())
        pcs_gt_list.append(pcs.detach().cpu())

    # End of training.
    torch.save(
        torch.vstack(pcs_recon_list).cpu(),
        f"{results_base_dir}/pcs_recon.pt",
    )
    torch.save(torch.vstack(pcs_gt_list).cpu(), f"{results_base_dir}/pcs_gt.pt")
    print(f"Saving screenshot to: {results_base_dir}/pcs_final.png")
    plot_recon_3d(
        pcs_recon_list[0],
        pcs_gt_list[0],
        num_pc=8,
        filename=f"{results_base_dir}/pcs_final.png",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for encoder training",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/encoder_airplane.yaml",
        type=str,
    )
    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile all the models",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Run a small subset.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Run a small subset.",
    )
    parser.add_argument(
        "--no-progressbar",
        default=False,
        action="store_true",
        help="Disable tqdm",
    )
    args = parser.parse_args()

    # Parse the args
    compile: bool = args.compile
    dev: bool = args.dev
    resume: bool = args.resume
    no_progressbar = args.no_progressbar

    (
        dataconfig,
        transformconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    ) = load_config(args.config_path)

    ##########################################################
    ### Inject dev.
    ##########################################################

    results_base_dir = "results"
    if dev:
        trainerconfig.max_epochs = 5000
        results_base_dir += "_dev"

    results_base_dir += f"/{loggerconfig.results_dir}"

    os.makedirs(f"{results_base_dir}", exist_ok=True)
    ##########################################################
    ### Load all assets
    ##########################################################

    fabric = Fabric(
        accelerator=trainerconfig.accelerator,
        precision=trainerconfig.precision,
    )

    seed_everything(trainerconfig.seed)

    # Logging
    logger = load_logger(loggerconfig)

    # Dataloaders
    dm = load_datamodule(dataconfig, dev=dev)
    dataloader = fabric.setup_dataloaders(dm.train_dataloader)
    valdataloader = fabric.setup_dataloaders(dm.test_dataloader)

    transform = load_transform(transformconfig)
    transform = fabric.setup_module(transform)

    # Create the model and dataset.
    model = load_model(modelconfig)

    print(summary(model))

    if resume:
        print(f"Resuming, loading model from: {results_base_dir}/model.ckpt")
        state = {"model": model}
        fabric.load(f"{results_base_dir}/model.ckpt", state)

    if compile:
        model = torch.compile(model)

    model.train()

    optimizer = Adam(
        model.parameters(),
        lr=modelconfig.learning_rate,
    )
    model, optimizer = fabric.setup(model, optimizer)

    # Loss ect transform

    loss_transform_config = TransformConfig(
        module="",
        ectconfig=modelconfig.ectlossconfig,
    )
    ect_transform_config = TransformConfig(
        module="",
        ectconfig=modelconfig.ectconfig,
    )
    losstransform = Transform(loss_transform_config)
    losstransform = fabric.setup_module(losstransform)

    ecttransform = Transform(ect_transform_config)
    ecttransform = fabric.setup_module(ecttransform)
    ##########################################################
    ### start the training.
    ##########################################################

    train(
        fabric,
        dataloader,
        transform,
        model,
        optimizer,
        losstransform,
        ecttransform,
        trainerconfig,
        loggerconfig,
        logger,
        no_progressbar,
        results_base_dir,
        dev,
    )
    test(
        fabric,
        valdataloader,
        transform,
        model,
        optimizer,
        losstransform,
        ecttransform,
        trainerconfig,
        loggerconfig,
        logger,
        no_progressbar,
        results_base_dir,
        dev,
    )


if __name__ == "__main__":
    main()
