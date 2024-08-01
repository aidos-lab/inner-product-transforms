"""
Input: 
- encoder config
- vae config 

output: 
- one row of the table: 
    mean chamfer distance between:
    - ground truth and ect recon
    - ground truth and vae recon 
    - ect recon and ground truth


    
Functions to implement 
- Chamfer distance 
- load models and data (mini check correct)
    - either encoder and vae or just encoder 
- loop over each element and compute the ect recon and vae recon 
- compute the chamfer distance for all three combinations
- add the objects to a running total 

- compute the mean of the losses in the test set.


Usage:
python evaluate_loss.py --encoder .\configs\config_encoder_mnist.yaml --vae .\configs\config_vae_mnist.yaml

"""

# NOTE: USE TORCH NO GRAD, OTHERWISE YOU END UP WITH OOM ERRORS FOR CUDA!


import torch
import argparse
from kaolin.metrics.pointcloud import chamfer_distance
from omegaconf import OmegaConf

from models.vae import VanillaVAE
from models.vae import BaseModel as BaseVAE

from models.encoder import BaseModel as EctEncoder

from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_loss_fn


from datasets import load_datamodule
from layers.ect import EctLayer, EctConfig

from layers.directions import generate_directions

from load_models import load_encoder, load_vae

DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 10, "c": ".5"}


def batched_chamfer_distance(true_batch, recon_batch, num_dims, num_pts):
    """
    Implementation of the chamfer distance for batched data in torch_geometric.
    Kaoling function only works on cuda so we hardcode the cuda.

    TODO: Fix the dimensions, this looks terrible.

    """
    batch_len = len(true_batch)
    if num_dims == 2:
        ch_dist = chamfer_distance(
            torch.cat(
                [
                    true_batch,
                    torch.zeros(size=(batch_len, num_pts, 1), device="cuda:0"),
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    recon_batch,
                    torch.zeros(size=(batch_len, num_pts, 1), device="cuda:0"),
                ],
                dim=-1,
            ),
        )
    else:
        ch_dist = chamfer_distance(
            true_batch,
            recon_batch,
        )

    return ch_dist


def evaluate_models(encoder_config, vae_config):
    dm = load_datamodule(encoder_config.data)
    encoder_model = load_encoder(encoder_config)
    vae_model = load_vae(vae_config=vae_config)

    total_ch_distance = {
        "encoder_ground_truth": [],
        "vae_ground_truth": [],
        "vae_encoder": [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dm.test_dataloader()):
            # Reconstruct the pointcloud using the encoder
            # model.
            batch = batch.to(DEVICE)
            ect = encoder_model.layer(batch, batch.batch)
            encoder_pointcloud = encoder_model(ect)

            # Reconstruct the ECT with the VAE model.
            # NOTE: Got some inperfect programming going on here,
            #       Hence the unsqueeze.
            reconstructed_ect, _, _, _ = vae_model(ect.unsqueeze(1))
            vae_pointcloud = encoder_model(reconstructed_ect)
            # # Print all the shapes, since they are a mess too.
            # # Need to reshape base point cloud.
            # print("batch", batch.x.view(-1, 256).shape)
            # print("encoder", encoder_pointcloud.shape)
            # print("vae", vae_pointcloud.shape)

            total_ch_distance["encoder_ground_truth"].append(
                batched_chamfer_distance(
                    batch.x.view(-1, 128, 2),
                    encoder_pointcloud.view(-1, 128, 2),
                    num_dims=2,
                    num_pts=128,
                )
            )

            total_ch_distance["vae_ground_truth"].append(
                batched_chamfer_distance(
                    batch.x.view(-1, 128, 2),
                    vae_pointcloud.view(-1, 128, 2),
                    num_dims=2,
                    num_pts=128,
                )
            )

            total_ch_distance["vae_encoder"].append(
                batched_chamfer_distance(
                    encoder_pointcloud.view(-1, 128, 2),
                    vae_pointcloud.view(-1, 128, 2),
                    num_dims=2,
                    num_pts=128,
                )
            )

    mean_ch_encoder_ground_truth = torch.hstack(
        total_ch_distance["encoder_ground_truth"]
    ).mean()
    mean_ch_vae_ground_truth = torch.hstack(
        total_ch_distance["vae_ground_truth"]
    ).mean()
    mean_ch_vae_encoder = torch.hstack(total_ch_distance["vae_encoder"]).mean()

    print(mean_ch_encoder_ground_truth)
    print(mean_ch_vae_ground_truth)
    print(mean_ch_vae_encoder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, help="Encoder configuration")
    parser.add_argument("--vae", type=str, help="VAE configuration")
    args = parser.parse_args()

    encoder_config = OmegaConf.load(args.encoder)
    vae_config = OmegaConf.load(args.vae)

    evaluate_models(encoder_config, vae_config)
