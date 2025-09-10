import argparse
import os

import torch
import torchvision
from lightning.fabric import Fabric
from loaders import load_config
from models.encoder_new import BaseLightningModel as Encoder
from models.schedulers.linear_scheduler import LinearNoiseScheduler
from models.unet import BaseLightningModel as Unet
from models.vqvae import BaseLightningModel as VQVAE
from plotting import plot_recon_3d
from torchvision.utils import make_grid
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):

    config, _ = load_config(args.unet_config)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.modelconfig.noise_scheduler)

    # Instantiate the model
    unet_model = Unet.load_from_checkpoint("trained_models/unet_airplane.ckpt").to(
        device
    )
    vae_model = VQVAE.load_from_checkpoint("trained_models/vqvae_airplane.ckpt").to(
        device
    )

    encoder_model = Encoder.load_from_checkpoint(
        "trained_models/encoder_new_airplane.ckpt"
    ).to(device)

    vae_model.eval()
    unet_model.eval()
    encoder_model.eval()

    xt = torch.randn((32, 9, 32, 32)).to(device)

    save_count = 0
    for i in tqdm(reversed(range(config.modelconfig.noise_scheduler.num_timesteps))):
        # for i in tqdm(reversed(range(10))):
        # Get prediction of noise
        noise_pred = unet_model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(
            xt, noise_pred, torch.as_tensor(i).to(device)
        )

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final iamge to save time
            ims = vae_model.model.decode(xt)

            pc = encoder_model(ims.squeeze())
            torch.save(pc, "results/unet/pc.pt")
            plot_recon_3d(
                pc.cpu().numpy(), pc.cpu().numpy(), filename="results/unet/recon.png"
            )

            ims = (1 + torch.clamp(ims, -1.0, 1.0).detach().cpu()) / 2
            grid = make_grid(ims, nrow=2)
            img = torchvision.transforms.ToPILImage()(grid[:3, :, :])
            img.save(f"results/unet/generated_ects_{i}.png")
            img.close()

        else:
            ims = xt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument(
        "--unet_config",
        dest="unet_config",
        default="configs/unet_airplane.yaml",
        type=str,
    )
    parser.add_argument(
        "--vae_config",
        dest="vae_config",
        default="configs/vqvae_airplane.yaml",
        type=str,
    )
    parser.add_argument(
        "--encoder_config",
        dest="encoder_config",
        default="configs/encoder_airplane.yaml",
        type=str,
    )
    args = parser.parse_args()
    main(args)
