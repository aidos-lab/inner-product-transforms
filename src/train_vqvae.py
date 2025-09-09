import argparse
import importlib
import pydantic
import yaml


import torch
from lightning import seed_everything
from lightning.fabric import Fabric
from lightning.pytorch.loggers import TensorBoardLogger
from src.datasets.shapenet import DataConfig
from src.loaders import (
    load_datamodule,
    load_logger,
    # load_transform,
    load_model,
)
from src.models.discriminator import Discriminator
from src.models.lpips import LPIPS
from torch.optim import Adam
from tqdm import tqdm

from src.datasets.transforms import EctTransform
from typing import Any

# Global settings.
torch.set_float32_matmul_precision("medium")


def load_module(config_dict: dict[Any, Any], classname: str) -> pydantic.BaseModel:
    module = importlib.import_module(config_dict["module"])
    config_class = getattr(module, classname)
    config = config_class(**config_dict)
    return config


def load_config(path: str):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict: dict[str, Any] = yaml.safe_load(stream)

    # Data
    dataconfig = load_module(config_dict["data"], classname="DataConfig")

    # Model
    modelconfig = load_module(config_dict["modelconfig"], classname="ModelConfig")

    # Trainer
    trainerconfig = load_module(config_dict["trainer"], classname="TrainerConfig")

    # Logger
    loggerconfig = load_module(config_dict["logger"], classname="LogConfig")

    return dataconfig, modelconfig, trainerconfig, loggerconfig


# def train(
#     config,
#     fabric,
#     dataloader,
#     model,
#     lpips_model,
#     discriminator,
#     optimizer_d,
#     optimizer_g,
#     recon_criterion,
#     disc_criterion,
#     ect_transform,
#     logger,
# ):
#     step_count = 0
#     for epoch in range(trainerconfig.epochs):
#         optimizer_g.zero_grad(set_to_none=False)
#         optimizer_d.zero_grad(set_to_none=True)
#
#         # Save model.
#         if epoch % trainerconfig.checkpoint_interval == 0:
#             state = {"model": model}
#             print(
#                 f"{trainerconfig.result_dir}/{trainerconfig.checkpoint}",
#             )
#             fabric.save(
#                 f"{trainerconfig.result_dir}/{trainerconfig.checkpoint}",
#                 state,
#             )
#
#         for pc in dataloader:
#             ect = ect_transform(pc).unsqueeze(1)
#
#             step_count += 1
#
#             # Start adding the discrimminator after 1k steps.
#             disc_scale_loss = 0
#             if step_count > 200:
#                 disc_scale_loss = 1
#
#             # Fetch autoencoders output(reconstructions)
#             output, _, quantize_losses = model(ect)
#
#             ######### Optimize Generator ##########
#             # L2 Loss
#             recon_loss = recon_criterion(output, ect)
#             g_loss = (
#                 recon_loss
#                 + (config.train.codebook_weight * quantize_losses["codebook_loss"])
#                 + (config.train.commitment_beta * quantize_losses["commitment_loss"])
#             )
#
#             # Adversarial loss only if disc_step_start steps passed
#             disc_fake_pred = discriminator(output)
#             disc_fake_loss = disc_criterion(
#                 disc_fake_pred,
#                 torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
#             )
#             g_loss += disc_scale_loss * config.train.disc_weight * disc_fake_loss
#
#             lpips_loss = torch.mean(lpips_model(output, ect))
#             g_loss += config.train.perceptual_weight * lpips_loss
#             fabric.backward(g_loss)
#             #####################################
#
#             ######### Optimize Discriminator #######
#             fake = output
#             disc_fake_pred = discriminator(fake.detach())
#             disc_real_pred = discriminator(ect)
#             disc_fake_loss = disc_criterion(
#                 disc_fake_pred,
#                 torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
#             )
#             disc_real_loss = disc_criterion(
#                 disc_real_pred,
#                 torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
#             )
#             disc_loss = config.train.disc_weight * (disc_fake_loss + disc_real_loss) / 2
#             fabric.backward(disc_loss)
#
#             optimizer_d.step()
#             optimizer_d.zero_grad(set_to_none=True)
#             optimizer_g.step()
#             optimizer_g.zero_grad()
#
#             # Logg metrics.
#             logger.log_metrics(
#                 {
#                     "disc_loss": disc_loss,
#                     "g_loss": g_loss,
#                     "recon_loss": recon_loss,
#                 },
#                 step=epoch,
#             )


def main(args):
    # Parse the args
    compile: bool = args.compile
    dev: bool = args.dev
    resume: bool = args.resume

    dataconfig, modelconfig, trainerconfig, loggerconfig = load_config(args.config_path)

    ##########################################################
    ### Inject dev.
    ##########################################################

    if dev:
        trainerconfig.max_epochs = 500
        loggerconfig.results_dir += "_dev"

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
    #
    # transform = load_transform(config.transform)
    #
    # # Create the model and dataset.
    # model = load_model(config.modelconfig)
    #
    # if resume:
    #     state = {"model": model}
    #     fabric.load("trained_models_dev/vqvae.ckpt", state)
    #
    # if compile:
    #     model = torch.compile(model)
    #
    # model.train()
    #
    # optimizer_g = Adam(
    #     model.parameters(),
    #     lr=trainerconfig.lr,
    #     betas=(0.5, 0.999),
    # )
    # model, optimizer_g = fabric.setup(model, optimizer_g)
    #
    # # Discrimminator
    # discriminator = Discriminator(config.discriminator)
    # if compile:
    #     discriminator = torch.compile(discriminator)
    #
    # optimizer_d = Adam(
    #     discriminator.parameters(),
    #     lr=trainerconfig.lr,
    #     betas=(0.5, 0.999),
    # )
    # discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)
    #
    # # LPIPS
    # # No need to freeze lpips as lpips.py takes care of that
    # lpips_model = LPIPS().eval()
    # lpips_model = fabric.setup(lpips_model)
    #
    # ############################################################
    # ### Loss functions
    # ############################################################
    #
    # # L1/L2 loss for Reconstruction
    # loss_fn_recon = torch.nn.MSELoss()
    # # Disc Loss can even be BCEWithLogits
    # loss_fn_disc = torch.nn.MSELoss()
    #
    # ##########################################################
    # ### start the training.
    # ##########################################################

    # train(
    #     config,
    #     fabric,
    #     dataloader,
    #     model,
    #     lpips_model,
    #     discriminator,
    #     optimizer_d,
    #     optimizer_g,
    #     loss_fn_recon,
    #     loss_fn_disc,
    #     transform,
    #     logger,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for vq vae training",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/vqvae_airplane_new.yaml",
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
    args = parser.parse_args()
    main(args)
