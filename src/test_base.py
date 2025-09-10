import argparse
import importlib
import json
import os

import torch

from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper
from models.encoder_new import BaseLightningModel as Encoder
from plotting import plot_ect, plot_recon_3d

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model: ModelWrapper, dm):
    m = dm.m.view(1, 1, 3).cuda()
    s = dm.s.squeeze().cuda()
    print("STD", s.shape, s)
    print("MEAN", m.shape)

    all_sample = []
    all_ref = []
    all_ects = []
    all_recon_ect = []
    for idx, pcs in enumerate(dm.val_dataloader):
        ect_gt = model.encoder.ect_transform(pcs.cuda())
        out_pc, reconstructed_ect = model.reconstruct(ect_gt.unsqueeze(1))

        out_pc = out_pc * s + m
        pcs = pcs * dm.s + dm.m
        all_sample.append(out_pc)
        all_ref.append(pcs)
        all_ects.append(ect_gt)
        all_recon_ect.append(reconstructed_ect)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    reconstructed_ect = torch.cat(all_recon_ect)
    ects = torch.cat(all_ects)

    # plot_ect(ects, reconstructed_ect, num_ects=5)
    cd_dist, emd_dist = EMD_CD(
        sample_pcs,
        ref_pcs,
        batch_size=100,
        reduced=True,
        accelerated_cd=True,
    )

    result = {"MMD-CD": cd_dist, "MMD-EMD": emd_dist}
    return result, sample_pcs, ref_pcs, reconstructed_ect, ects


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
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    parser.add_argument(
        "--num_reruns",
        default=1,
        type=int,
        help="Number of reruns for the standard deviation.",
    )
    args = parser.parse_args()

    ##################################################################
    ### Encoder
    ##################################################################
    encoder_config, _ = load_config(args.encoder_config)

    encoder_model = Encoder.load_from_checkpoint(
        "trained_models/encoder_new_airplane.ckpt"
    ).cuda()

    dm = load_datamodule(encoder_config.data, dev=args.dev)

    ects_gt = torch.load("results/vae_baseline_airplane/gt_ect.pt").cuda()
    ects_recon = torch.load("results/vae_baseline_airplane/recon_ect.pt").cuda()
    ects_sample = torch.load("results/vae_baseline_airplane/sample_ect.pt").cuda()

    pointcloud_gt = encoder_model(ects_gt.squeeze()).view(
        -1, encoder_config.modelconfig.num_pts, 3
    )
    pointcloud_recon = encoder_model(ects_recon.squeeze()).view(
        -1, encoder_config.modelconfig.num_pts, 3
    )
    pointcloud_sample = encoder_model(ects_sample.squeeze()).view(
        -1, encoder_config.modelconfig.num_pts, 3
    )

    plot_recon_3d(
        pointcloud_gt.detach().cpu().numpy(), pointcloud_recon.detach().cpu().numpy()
    )

    plot_recon_3d(
        pointcloud_gt.detach().cpu().numpy(), pointcloud_sample.detach().cpu().numpy()
    )
