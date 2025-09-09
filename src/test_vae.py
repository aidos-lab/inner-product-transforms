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
        "--dev",
        default=False,
        action="store_true",
        help="Only evaluate on the first batch.",
    )
    args = parser.parse_args()

    ##################################################################
    ### Encoder
    ##################################################################

    # Load the datamodule
    # NOTE: Loads the datamodule from the encoder and does not check for
    # equality of the VAE data configs.

    vae_config, _ = load_config(args.vae_config)

    if args.dev:
        vae_config.trainer.save_dir += "_dev"

    model = load_model(
        vae_config.modelconfig,
        f"./{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
    ).to(DEVICE)

    model_name = vae_config.trainer.model_name.split(".")[0]
    dm = load_datamodule(vae_config.data, dev=args.dev)

    results = []

    # Evaluate reconstruction
    ref_pcs, reconstructed_ect, ects = evaluate_reconstruction(model, dm)

    #####################################################
    ### Saving and printing
    #####################################################

    result_suffix = ""
    if args.dev:
        result_suffix = "_dev"

    # Make sure folders exist
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results_dev", exist_ok=True)

    # Add some samples
    sample_ect = model.model.sample(64)

    torch.save(
        sample_ect,
        f"./results{result_suffix}/{model_name}/sample_ect.pt",
    )

    torch.save(
        reconstructed_ect,
        f"./results{result_suffix}/{model_name}/reconstructed_ect.pt",
    )
    torch.save(
        ects,
        f"./results{result_suffix}/{model_name}/ref_ect.pt",
    )
    torch.save(
        ref_pcs,
        f"./results{result_suffix}/{model_name}/ref_pcs.pt",
    )
