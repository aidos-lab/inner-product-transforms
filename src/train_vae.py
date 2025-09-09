import argparse
import importlib
import os
from typing import Any

import pydantic
import torch
import torchvision
import yaml
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from torchvision.utils import make_grid
from tqdm import tqdm

from src.loaders import load_datamodule, load_logger, load_model, load_transform
from src.models.discrimminator import Discrimminator, DiscrimminatorConfig
from src.models.lpips import LPIPS

# Global settings.
torch.set_float32_matmul_precision("medium")


def load_module(config_dict: dict[Any, Any], classname: str) -> pydantic.BaseModel:

    module_name = config_dict.get("module", None)

    if module_name is not None:
        module = importlib.import_module(config_dict["module"])
        config_class = getattr(module, classname)
        config = config_class(**config_dict)
    else:
        module = importlib.import_module("__main__")
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

    # Transform
    transformconfig = load_module(config_dict["transform"], classname="TransformConfig")

    # Model
    modelconfig = load_module(config_dict["modelconfig"], classname="ModelConfig")

    # Trainer
    trainerconfig = load_module(config_dict["trainer"], classname="TrainerConfig")

    # Logger
    loggerconfig = load_module(config_dict["logger"], classname="LogConfig")

    return dataconfig, transformconfig, modelconfig, trainerconfig, loggerconfig


def train(
    trainerconfig,
    modelconfig,
    loggerconfig,
    fabric,
    dataloader,
    valdataloader,
    model,
    lpips_model,
    discrimminator,
    optimizer_d,
    optimizer_g,
    recon_criterion,
    disc_criterion,
    transform,
    logger,
    no_progressbar,
):
    step_count = 0
    for epoch in range(trainerconfig.max_epochs):
        optimizer_g.zero_grad(set_to_none=False)
        optimizer_d.zero_grad(set_to_none=True)

        # Save model.
        if epoch % trainerconfig.checkpoint_interval == 0:
            state = {"model": model}
            print(
                f" Saving model to results/{loggerconfig.results_dir}/model.ckpt",
            )
            fabric.save(
                f"results/{loggerconfig.results_dir}/model.ckpt",
                state,
            )

        for pc in tqdm(dataloader, disable=no_progressbar):
            ect = transform(pc).unsqueeze(1)

            step_count += 1

            # Start adding the discrimminator after 1k steps.
            disc_scale_loss = 0
            if step_count > 200:
                disc_scale_loss = 1

            # Fetch autoencoders output(reconstructions)
            output, internal_loss, quant_losses = model(ect)

            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, ect)
            g_loss = recon_loss + internal_loss

            # Adversarial loss only if disc_step_start steps passed
            disc_fake_pred = discrimminator(output)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
            )
            g_loss += disc_scale_loss * modelconfig.disc_weight * disc_fake_loss

            lpips_loss = torch.mean(lpips_model(output, ect))
            g_loss += modelconfig.perceptual_weight * lpips_loss
            fabric.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            fake = output
            disc_fake_pred = discrimminator(fake.detach())
            disc_real_pred = discrimminator(ect)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
            )
            disc_real_loss = disc_criterion(
                disc_real_pred,
                torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
            )
            disc_loss = modelconfig.disc_weight * (disc_fake_loss + disc_real_loss) / 2
            fabric.backward(disc_loss)

            optimizer_d.step()
            optimizer_d.zero_grad(set_to_none=True)
            optimizer_g.step()
            optimizer_g.zero_grad()

            # Logg metrics.
            logger.log_metrics(
                {
                    "disc_loss": disc_loss,
                    "g_loss": g_loss,
                    "recon_loss": recon_loss,
                }
                | quant_losses,
                step=epoch,
            )

    os.makedirs(name=f"results/{loggerconfig.results_dir}/test", exist_ok=True)
    # Starting the test loop.
    recon = []
    gt = []
    for idx, pc in enumerate(valdataloader):
        ect = transform(pc).unsqueeze(1)
        gt.append(ect)
        # Fetch autoencoders output(reconstructions)
        output, internal_loss, quant_losses = model(ect)
        recon.append(output)

        if idx == 0:
            sample_size = 8
            save_recon = (
                1 + torch.clamp(output[:sample_size], -1.0, 1.0).detach().cpu()
            ) / 2
            save_gt = ((ect[:sample_size] + 1) / 2).detach().cpu()

            grid = make_grid(torch.cat([save_gt, save_recon], dim=0), nrow=sample_size)
            img = torchvision.transforms.ToPILImage()(grid)
            img.save(f"results/{loggerconfig.results_dir}/ect_recon.png")
            img.close()
            # if idx == 0:
            #     logger.log_image(
            #         "generated_images",
            #         [
            #             grid,
            #         ],
            #         0,
            #     )


def main():
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

    if dev:
        trainerconfig.max_epochs = 10
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
    valdataloader = fabric.setup_dataloaders(dm.test_dataloader)

    transform = load_transform(transformconfig)
    transform = fabric.setup_module(transform)

    # Create the model and dataset.
    model = load_model(modelconfig)

    if resume:
        state = {"model": model}
        fabric.load(f"results/{loggerconfig.results_dir}/model.ckpt", state)

    if compile:
        model = torch.compile(model)

    model.train()

    optimizer_g = Adam(
        model.parameters(),
        lr=modelconfig.lr,
        betas=(0.5, 0.999),
    )
    model, optimizer_g = fabric.setup(model, optimizer_g)

    # Discrimminator
    discrimminatorconfig = DiscrimminatorConfig(
        im_channels=modelconfig.im_channels,
        lr=0.0001,
    )
    discrimminator = Discrimminator(discrimminatorconfig)
    if compile:
        discrimminator = torch.compile(discrimminator)

    optimizer_d = Adam(
        discrimminator.parameters(),
        lr=discrimminatorconfig.lr,
        betas=(0.5, 0.999),
    )
    discrimminator, optimizer_d = fabric.setup(discrimminator, optimizer_d)

    # LPIPS
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval()
    lpips_model = fabric.setup(lpips_model)

    ############################################################
    ### Loss functions
    ############################################################

    # L1/L2 loss for Reconstruction
    loss_fn_recon = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    loss_fn_disc = torch.nn.MSELoss()

    ##########################################################
    ### start the training.
    ##########################################################

    train(
        trainerconfig,
        modelconfig,
        loggerconfig,
        fabric,
        dataloader,
        valdataloader,
        model,
        lpips_model,
        discrimminator,
        optimizer_d,
        optimizer_g,
        loss_fn_recon,
        loss_fn_disc,
        transform,
        logger,
        no_progressbar,
    )


if __name__ == "__main__":
    main()
