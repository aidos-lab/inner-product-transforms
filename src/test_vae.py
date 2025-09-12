"""Generates some images and saves them"""

import argparse
import json
import os

import torch

from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper
from plotting import plot_ect, plot_recon_3d

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model, dm):
    all_ref = []
    all_ects = []
    all_recon_ect = []
    for _, pcs in enumerate(dm.val_dataloader):
        pcs = pcs[0]
        ect_gt = model.ect_transform(pcs.cuda())
        reconstructed_ect, _, _, _ = model.model(ect_gt)

        all_ref.append(pcs)
        all_ects.append(ect_gt)
        all_recon_ect.append(reconstructed_ect)

    ref_pcs = torch.cat(all_ref, dim=0)
    reconstructed_ect = torch.cat(all_recon_ect)
    ects = torch.cat(all_ects)

    return ref_pcs, reconstructed_ect, ects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae_config",
        required=True,
        default=None,
        type=str,
        help="VAE Configuration",
    )
    parser.add_argument(
        "--encoder_config",
        required=True,
        default=None,
        type=str,
        help="VAE Configuration",
    )
    args = parser.parse_args()

    ##################################################################
    ### Encoder
    ##################################################################

    # Load the datamodule
    # NOTE: Loads the datamodule from the encoder and does not check for
    # equality of the VAE data configs.

    # encoder_config = load_config(args.encoder_config)
    #
    # # Third argument is the model config.
    # encoder_model = load_model(encoder_config[2])
    # encoder_model.eval()

    (
        dataconfig,
        transformconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    ) = load_config(args.vae_config)

    results = []

    # Load the saved tensors.

    ect_recon = torch.load(f"results/{loggerconfig.results_dir}/recon_ect.pt")
    ect_gt = torch.load(f"results/{loggerconfig.results_dir}/gt_ect.pt")
    pcs_gt = torch.load(f"results/{loggerconfig.results_dir}/gt_pcs.pt")

    #####################################################
    ### Evaluate
    #####################################################

    # with torch.no_grad():
    #     pcs_recon = encoder_model(ect_recon)
    #
    # result = EMD_CD(pcs_recon, pcs_gt, batch_size=128, accelerated_cd=True)
    #
    # print(result)
