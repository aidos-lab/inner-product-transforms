"""
Evaluation script for the generated ECT's from the
VAE. We calculate the FID Score between the generated
images and the training set and the test set.
"""

import argparse
import json
import sys

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize


def pre_validation_ect(ect):
    """
    Prints sanity checks to make sure the model input is what we expect.
    The ECT has to take on values in the interval [0,1].
    """

    assert torch.allclose(ect.min(), torch.tensor(0))
    assert torch.allclose(ect.max(), torch.tensor(1))


def preprocess_ect(ect):
    """
    Preprocesses the ECT sets to be compatible with the
    FID, which assumes an image tensor of shape [N,3,299,299]
    with dtype uint 8.
    To that end we resize the ECT and convert to
    """
    t = Resize(299)
    resized_ect = t(255 * ect).to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1)
    return resized_ect


def compute_ect_fid(sample_ect, reference_ect):
    """
    Calculates the FID for two sets of ECTs.
    The ECT tensors are assumed to be of shape [N,3,299,299],
    where N is the number of ECTs with dtype uint
    """
    fid = FrechetInceptionDistance(feature=64)
    for i in range(0, len(sample_ect), 64):
        fid.update(sample_ect[i : i + 64], real=True)
        fid.update(reference_ect[i : i + 64], real=False)
    return fid.compute()


def evaluate_fid(sample_ect, reference_ect):
    samples = preprocess_ect(sample_ect)
    references = preprocess_ect(reference_ect)

    f = compute_ect_fid(samples, references)

    return f


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_folder",
        type=str,
        help="Input folder under results, e.g. ./results/vae_airplane",
    )
    parser.add_argument(
        "--reference_ect",
        type=str,
        default=None,
        help="Reference ects of the training set.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=None,
        help="Reference ects of the training set.",
    )
    args = parser.parse_args()

    results_folder = args.results_folder

    # Hook into the dev flag and point to the dev results folder.
    if args.dev:
        pass

    generated_ect_file = torch.load(
        f"{results_folder}/sample_ect.pt", weights_only=True
    ).cpu()
    reconstructed_ect_file = torch.load(
        f"{results_folder}/reconstructed_ect.pt", weights_only=True
    ).cpu()
    validation_ect_file = torch.load(
        f"{results_folder}/ect.pt", weights_only=True
    ).cpu()

    result = {
        "generated_vs_reconstructed_fid": evaluate_fid(
            generated_ect_file, reconstructed_ect_file
        ).item(),
        "reconstructed_vs_validation_fid": evaluate_fid(
            reconstructed_ect_file, validation_ect_file
        ).item(),
        "generated_vs_validation_fid": evaluate_fid(
            generated_ect_file, validation_ect_file
        ).item(),
        "generated_vs_generated_fid": evaluate_fid(
            generated_ect_file, generated_ect_file
        ).item(),
    }

    with open(f"{args.results_folder}/generation_evaluation.json", "w") as f:
        json.dump(result, f, indent=4)
    print(json.dumps(result, indent=4))
