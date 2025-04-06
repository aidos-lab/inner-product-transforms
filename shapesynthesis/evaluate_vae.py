"""
Evaluation script for the generated ECT's from the
VAE. We calculate the FID Score between the generated
images and the training set and the test set.
"""

import argparse
import json

import torch
from PIL.Image import re
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize

# # Load the images into memory from the results folder.
# # The "latent" postfix are the results used in the paper.
# sample_ect = torch.load("../results/vae_airplane_latent/sample_ect.pt").cpu()
# reference_ect = torch.load("../results/vae_airplane_latent/ect.pt").cpu()


def pre_validation_ect(sample_ect, reference_ect):
    """
    Prints sanity checks to make sure the model input is what we expect.
    The ECT has to take on values in the interval [0,1].
    """

    # Check for shape equal
    assert sample_ect.shape == reference_ect.shape

    # Stats for the reference
    print("Statistics reference ects.")
    print(72 * "=")
    print(reference_ect.min().item(), reference_ect.max().item())
    print("Statistics on the generated ects.")
    print(sample_ect.min().item(), sample_ect.max().item())
    print(reference_ect.shape, sample_ect.shape)


def post_validation_ect(sample_ect, reference_ect):
    """
    Prints sanity checks to make sure the model input is what we expect.
    """

    # Stats for the reference
    # print(72 * "=")
    print("Statistics on the reference ects.")
    # print(72 * "=")
    print(
        "Min:",
        reference_ect.min().item(),
        "Max:",
        reference_ect.max().item(),
        "dtype",
        reference_ect.dtype,
    )

    # print(72 * "=")
    print("Statistics on the generated ects.")
    # print(72 * "=")
    print(
        "Min:",
        sample_ect.min().item(),
        "Max:",
        sample_ect.max().item(),
        "dtype",
        reference_ect.dtype,
    )


def preprocess_ect(sample_ect, reference_ect):
    """
    Preprocesses the ECT sets to be compatible with the
    FID, which assumes an image tensor of shape [N,3,299,299]
    with dtype uint 8.
    To that end we resize the ECT and convert to
    """
    t = Resize(299)
    resized_sample_ect = (
        t(255 * sample_ect).to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1)
    )
    resized_reference_ect = (
        t(255 * reference_ect).to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1)
    )
    return resized_sample_ect, resized_reference_ect


def compute_ect_fid(sample_ect, reference_ect):
    """
    Calculates the FID for two sets of ECTs.
    The ECT tensors are assumed to be of shape [N,3,299,299],
    where N is the number of ECTs with dtype uint
    """
    fid = FrechetInceptionDistance(feature=64)
    fid.update(sample_ect[:300], real=True)
    fid.update(reference_ect[:300], real=False)

    return fid.compute()


def evaluate_fid(sample_ect_file, reference_ect_file):
    sample_ect = torch.load(sample_ect_file).cpu()
    reference_ect = torch.load(reference_ect_file).cpu()

    pre_validation_ect(sample_ect, reference_ect)
    processed_samples, processed_references = preprocess_ect(sample_ect, reference_ect)
    post_validation_ect(processed_samples, processed_references)
    f = compute_ect_fid(processed_samples, processed_references)

    return f


if __name__ == "__main__":

    """
    Simple argparser for the results folder.
    The folder is the folder under results, e.g. ./results/vae_airplane
    (without a trailing slash).
    We assume that the following files are present:
    - ect.pt
        The ect of the reference point cloud
    - sample_ect.pt
        The generated ects by the model.
    - reconstructed_ect.pt
        The reconstructed ECTs by the VAE.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "INPUT",
        type=str,
        help="Input folder under results, e.g. ./results/vae_airplane",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Reference ects",
    )
    args = parser.parse_args()

    sample_ect_file = f"{args.INPUT}/sample_ect.pt"
    recon_ect_file = f"{args.INPUT}/reconstructed_ect.pt"
    ref_ect_file = f"{args.INPUT}/ect.pt"

    result = {
        "sample_vs_validation_fid": evaluate_fid(sample_ect_file, ref_ect_file).item(),
        "reconstruction_vs_validation_fid": evaluate_fid(
            recon_ect_file, ref_ect_file
        ).item(),
    }

    if args.train is not None:
        train_result = {
            "sample_vs_train_fid": evaluate_fid(sample_ect_file, args.train).item(),
            "reconstruction_vs_train_fid": evaluate_fid(
                recon_ect_file, args.train
            ).item(),
            "validation_vs_train_fid": evaluate_fid(ref_ect_file, args.train).item(),
        }
        result |= train_result

    print(json.dumps(result, indent=4))
