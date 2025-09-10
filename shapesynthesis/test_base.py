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

    # Inject dev runs if needed.
    if args.dev:
        encoder_config.trainer.save_dir += "_dev"

    print(
        f"Loading model from ./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}"
    )
    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)
    encoder_model.model.eval()

    # Set model name for saving results in the results folder.
    model_name = encoder_config.trainer.model_name.split(".")[0]

    dm = load_datamodule(encoder_config.data, dev=args.dev)

    # Load the datamodule
    # NOTE: Loads the datamodule from the encoder and does not check for
    # equality of the VAE data configs.

    if args.vae_config:
        vae_config, _ = load_config(args.vae_config)

        if args.dev:
            vae_config.trainer.save_dir += "_dev"

        # Check that the configs are equal for the ECT.
        assert vae_config.ectconfig == encoder_config.ectconfig

        vae_model = load_model(
            vae_config.modelconfig,
            f"./{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
        ).to(DEVICE)
        print(
            f"Loading vae model from ./{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
        )

        # If VAE is provided, overwrite the modelname.
        model_name = vae_config.trainer.model_name.split(".")[0]
    else:
        vae_model = None

    encoder_model.eval()

    model = ModelWrapper(encoder_model, vae_model)

    ects_gt = torch.load("results/vae_baseline_airplane/gt_ect.pt").cuda()
    ects_recon = torch.load("results/vae_baseline_airplane/recon_ect.pt").cuda()
    print(ects_gt.shape)

    pointcloud_gt = encoder_model(ects_gt.squeeze()).view(
        -1, encoder_config.modelconfig.num_pts, 3
    )
    pointcloud_recon = encoder_model(ects_recon.squeeze()).view(
        -1, encoder_config.modelconfig.num_pts, 3
    )

    plot_recon_3d(
        pointcloud_gt.detach().cpu().numpy(), pointcloud_recon.detach().cpu().numpy()
    )
