import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F

import numpy as np

# from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD

# from model_wrapper import ModelNoOpWrapper, ModelWrapper
from datasets.shapenet_data_pc import ShapeNet15kPointClouds

# from datasets.shapenetcore import get_datasets
from layers.ect import compute_ect_point_cloud
from layers.directions import generate_uniform_directions
from models.encoder import BaseLightningModel as Encoder
from models.vae_1d import BaseLightningModel as VAE

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_datasets(category, dataroot, npoints=2048):
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
        random_subsample=False,
    )
    return te_dataset


class ModelWrapper:
    def __init__(self, encoder: Encoder, vae: VAE | None = None) -> None:
        self.encoder = encoder
        self.config = encoder.config
        self.v = generate_uniform_directions(
            num_thetas=self.config.ectconfig.num_thetas,
            d=self.config.ectconfig.ambient_dimension,
            seed=self.config.ectconfig.seed,
        ).cuda()
        self.vae = vae
        if vae:
            self.vae.eval()

    @torch.no_grad()
    def sample(self, num_samples: int, num_points: int, ambient_dimension: int):
        """
        out_pc, sample_ect = model.sample(len(batch), pc_shape)
        The way we expect the input
        B is the number of point clouds
        N is the number of points per cloud.
        _, out_pc = model.sample(B, N)
        """
        ect_samples = self.vae.model.sample(
            num_samples=num_samples, device="cuda:0"
        ).squeeze()

        # Rescale to 0,1
        ect_samples = (ect_samples + 1) / 2

        print(ect_samples.shape)

        vae_pointcloud = self.encoder(ect_samples).view(
            num_samples, self.encoder.config.num_pts, ambient_dimension
        )
        return vae_pointcloud, ect_samples

    @torch.no_grad()
    def reconstruct(self, batch):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """
        pc_shape = batch["train_points"].shape
        tr_pc = batch["train_points"].cuda()
        reconstructed_ect = None
        ect = (
            compute_ect_point_cloud(
                x=tr_pc,
                v=self.v,
                radius=self.config.ectconfig.r,
                resolution=self.config.ectconfig.resolution,
                scale=64,
            )
            / 2048
        )

        if self.vae is not None:
            # print(self.vae)
            # Rescale to [-1,1] for the VAE

            ect = 2 * ect.unsqueeze(1) - 1
            # print(ect.shape)
            # [self.decode(z), input_tensor, mu, log_var]
            reconstructed_ect, _, _, _ = self.vae.model(ect)

            # Rescale to 0,1
            reconstructed_ect = (reconstructed_ect.squeeze() + 1) / 2
            pointcloud = self.encoder(reconstructed_ect).view(
                -1, self.encoder.config.num_pts, pc_shape[-1]
            )
            # assert torch.allclose(reconstructed_ect, batch.ect)
        else:
            pointcloud = self.encoder(ect).view(
                -1, self.encoder.config.num_pts, pc_shape[-1]
            )

        return pointcloud, reconstructed_ect, ect


@torch.no_grad()
def evaluate_reconstruction(model: ModelWrapper, encoder_config):
    # dm = load_datamodule(encoder_config.data)
    te_ds = get_datasets(
        encoder_config.data.cates,
        "./data/shapenet/raw/ShapeNetCore.v2.PC15k",
    )
    loader = torch.utils.data.DataLoader(te_ds, batch_size=8)
    all_sample = []
    all_ref = []
    all_means = []
    all_std = []
    all_ects = []
    all_recon_ect = []
    for i, batch in enumerate(loader):
        out_pc, reconstructed_ect, ect = model.reconstruct(batch)
        pc_shape = batch["train_points"].shape

        m, s = batch["mean"].float().cuda(), batch["std"].float().cuda()

        te_pc = batch["test_points"].cuda()
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

        if torch.is_tensor(reconstructed_ect):
            all_ects.append(ect)
            all_recon_ect.append(reconstructed_ect)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    if torch.is_tensor(reconstructed_ect):
        reconstructed_ect = torch.cat(all_recon_ect)
        ects = torch.cat(all_ects)
    else:
        reconstructed_ect = None
        ects = None

    if pc_shape[-1] == 2:
        sample_pcs = F.pad(
            input=sample_pcs, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0
        )
        ref_pcs = F.pad(input=ref_pcs, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)

    results = EMD_CD(
        sample_pcs,
        ref_pcs,
        batch_size=8,
        reduced=True,
        accelerated_cd=True,
    )

    if pc_shape[-1] == 2:
        sample_pcs = sample_pcs[:, :, :2]
        ref_pcs = ref_pcs[:, :, :2]

    results = {
        ("%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in results.items()
    }

    return results, sample_pcs, ref_pcs, reconstructed_ect, ects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_config",
        required=True,
        type=str,
        help="Encoder configuration",
    )
    parser.add_argument(
        "--vae_config",
        required=False,
        default=None,
        type=str,
        help="VAE Configuration (Optional)",
    )
    parser.add_argument(
        "--num_reruns",
        default=1,
        type=int,
        help="Number of reruns for the standard deviation.",
    )
    parser.add_argument(
        "--generative",
        default=False,
        action="store_true",
        help="Evaluation generative performance.",
    )
    parser.add_argument(
        "--noop",
        default=False,
        action="store_true",
        help="Bypass model and evaluate training points vs test points.",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="Bypass model and evaluate training points vs test points.",
    )
    args = parser.parse_args()

    encoder_config, _ = load_config(args.encoder_config)

    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)
    encoder_model.model.eval()

    # Set model name for saving results in the results folder.
    model_name = encoder_config.trainer.model_name.split(".")[0]

    # # dm = load_datamodule(encoder_config.data)
    # te_ds = get_datasets(
    #     encoder_config.data.cates,
    #     "./data/shapenet/raw/ShapeNetCore.v2.PC15k",
    # )
    # loader = torch.utils.data.DataLoader(te_ds, batch_size=8)
    # Load the datamodule
    # NOTE: Loads the datamodule from the encoder and does not check for
    # equality of the VAE data configs.

    if args.vae_config:
        vae_config, _ = load_config(args.vae_config)

        # Check that the configs are equal for the ECT.
        assert vae_config.ectconfig == encoder_config.ectconfig

        vae_model = load_model(
            vae_config.modelconfig,
            f"./{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
        ).to(DEVICE)

        # If VAE is provided, overwrite the modelname.
        model_name = vae_config.trainer.model_name.split(".")[0]
    else:
        vae_model = None

    encoder_model.eval()

    # # Overwrite model if noop is passed
    # if args.noop:
    #     model = ModelNoOpWrapper()
    # else:
    model = ModelWrapper(encoder_model, vae_model)

    # for i, batch in enumerate(dm.test_dataloader()):
    #     out_pc = model.reconstruct(batch.to(DEVICE))
    #     break

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, recon_ect, ects = evaluate_reconstruction(
            model, encoder_config
        )
        result["normalized"] = args.normalize
        result["model"] = model_name

        if args.normalize:
            suffix = "_normalized"
        else:
            suffix = ""

        results.append(result)

    # Save the results in json format, {config name}.json
    # Example ./results/encoder_mnist.json
    with open(
        f"./results/{model_name}/{model_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)
    torch.save(
        sample_pc,
        f"./results/{model_name}/reconstructions{suffix}.pt",
    )
    torch.save(
        ref_pc,
        f"./results/{model_name}/references{suffix}.pt",
    )
    torch.save(
        recon_ect,
        f"./results/{model_name}/reconstructed_ect.pt",
    )
    torch.save(
        ects,
        f"./results/{model_name}/ect.pt",
    )
    pprint(results)
