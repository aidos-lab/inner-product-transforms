"""
Simple argparser for the results folder.
The folder is the folder under results, e.g. ./results/vae_airplane
(without a trailing slash).
We assume that the following files are present:
- references.pt
    Reference point clouds from the validation set.
- reconstructions.pt
    Reconstructed point clouds from the VAE (or Encoder)

Optional files:
- samples.pt
    File with the generated point clouds.
"""

import argparse
import json
from pathlib import Path

import torch

from shapesynthesis.metrics.evaluation import compute_all_metrics


# Custom serializer for Tensors in json
def tensor_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_folder",
    type=str,
    help="Input folder under results, e.g. ./results/vae_airplane",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Number of samples to use",
)

args = parser.parse_args()

# Check if the results folder exists and is a folder.
results_folder = Path(args.results_folder)
if not results_folder.is_dir():
    raise ValueError("Results folder not correct, please check the path you provided.")

# Check if results file exists in the folder.
# Overwrite the empty results dictionary with it.
results_file = Path(f"{results_folder}/results.json")
results = {}
if results_file.is_file():
    with open(results_file, "r") as f:
        results = json.load(f)

reconstructed_point_cloud_file = Path(f"{args.results_folder}/reconstructions.pt")
validation_point_cloud_file = Path(f"{args.results_folder}/references.pt")
generated_point_cloud_file = Path(f"{args.results_folder}/samples.pt")

# Load all the files in memory.
reconstructed = torch.load(reconstructed_point_cloud_file)
validation = torch.load(validation_point_cloud_file)

if args.num_samples is not None:
    reconstructed = reconstructed[: args.num_samples]
    validation = validation[: args.num_samples]

# Merge (update keys) with updated metrics.
results |= {
    "reconstructed_vs_validation_nna": compute_all_metrics(
        reconstructed, validation, batch_size=100
    ),
}

# Check if the generated point cloud exists.
if generated_point_cloud_file.is_file():
    generated = torch.load(generated_point_cloud_file)
    if args.num_samples is not None:
        generated = generated[: args.num_samples]

    # Compute extra statistics
    results |= {
        "reconstructed_vs_validation_nna": compute_all_metrics(
            reconstructed, validation, batch_size=100
        ),
    }

print(results)

# Write it back to the results folder into the results file.
with open(f"{args.results_folder}/results.json", "w") as f:
    json.dump(results, f, indent=4, default=tensor_serializer)
print(json.dumps(results, indent=4, default=tensor_serializer))
