import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F

import numpy as np

from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD

# from model_wrapper import ModelCompletionWrapper
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from layers.ect import compute_ect_point_cloud
from layers.directions import generate_uniform_directions
from models.encoder import BaseLightningModel as Encoder

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataset(dataroot, npoints, category):
    tr_dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=category,
        split="train",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True,
    )
    te_dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=category,
        split="val",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        random_subsample=True,
    )
    return te_dataset


class ModelCompletionWrapper:
    def __init__(
        self,
        encoder: Encoder,
    ) -> None:
        self.encoder = encoder.eval()
        self.config = encoder.config

        # Sample 200 points
        self.subset_indexes = torch.randperm(n=self.config.num_pts)[:200]
        self.v = generate_uniform_directions(
            num_thetas=self.config.ectconfig.num_thetas,
            d=self.config.ectconfig.ambient_dimension,
            seed=self.config.ectconfig.seed,
        ).cuda()

    @torch.no_grad()
    def reconstruct(self, batch, normalized=False):
        """
        The method first subsamples the point cloud to create a point cloud that
        it can recreate. The completion model has _not_ been finetuned and
        consists of the standard encoder. The ECT's it is decoding in this
        experiment are thus far out of distribution.
        """
        tr_pcs = batch["train_points"].cuda()

        pc_shape = tr_pcs.shape

        # print(batch.x.shape)
        # Select 200 random point from the dataset.
        sparse_pointcloud = tr_pcs.view(-1, 2048, 3)[:, self.subset_indexes, :]

        print("computeing ect")
        print(sparse_pointcloud.shape)
        sparse_ect = (
            compute_ect_point_cloud(
                x=sparse_pointcloud,
                v=self.v,
                radius=self.config.ectconfig.r,
                resolution=self.config.ectconfig.resolution,
                scale=self.config.ectconfig.scale,
            )
            / 200
        )

        # Complete the point cloud to 2048 points.
        pointcloud = self.encoder(sparse_ect).view(
            -1, self.encoder.config.num_pts, pc_shape[-1]
        )

        return pointcloud, None


@torch.no_grad()
def evaluate_reconstruction(model, loader):
    all_sample = []
    all_ref = []
    all_means = []
    all_std = []

    for i, batch in enumerate(loader):

        out_pc, _ = model.reconstruct(batch)
        pc_shape = batch["train_points"]

        m, s = batch["mean"].float().cuda(), batch["std"].float().cuda()

        te_pc = batch["train_points"].cuda()
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    results = EMD_CD(
        sample_pcs,
        ref_pcs,
        batch_size=8,
        reduced=True,
        accelerated_cd=True,
    )

    results = {
        ("%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in results.items()
    }

    return (results, sample_pcs, ref_pcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_completion_config",
        required=True,
        type=str,
        help="Encoder configuration",
    )

    parser.add_argument(
        "--num_reruns",
        default=1,
        type=int,
        help="Number of reruns for the standard deviation.",
    )

    args = parser.parse_args()

    encoder_config, _ = load_config(args.encoder_completion_config)

    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)

    # Set model name for saving results in the results folder.
    model_name_stem = encoder_config.trainer.model_name.split(".")[0].split("_")
    model_name = model_name_stem[0] + "_completion_" + model_name_stem[1]

    # Load data
    # dm = load_datamodule(encoder_config.data)
    # get_dataset(dataroot, npoints,category):
    te_ds = get_dataset(
        "/mnt/c/Users/ernst/Documents/02-Work/06-Repos/PVD/ShapeNetCore.v2.PC15k",
        2048,
        encoder_config.data.cates,
    )
    loader = torch.utils.data.DataLoader(te_ds, batch_size=64)

    model = ModelCompletionWrapper(encoder_model)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc = evaluate_reconstruction(model, loader)
        result["normalized"] = False
        result["model"] = model_name

        suffix = ""

        results.append(result)

    # Example ./results/encoder_mnist.json
    with open(
        f"./results/{model_name}/{model_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)
    pprint(results)
