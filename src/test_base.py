import argparse
import importlib
import json
import os

import matplotlib.pyplot as plt
import torch

from loaders import load_config, load_datamodule
from metrics.evaluation import EMD_CD

# from model_wrapper import ModelWrapper
from plotting import plot_ect, plot_recon_3d

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(config, model_path=None):
    module = importlib.import_module(config.module)
    model_class = getattr(module, "BaseLightningModel")

    if model_path:
        model = model_class.load_from_checkpoint(model_path)
    else:
        config_dict = json.loads(json.dumps(config, default=lambda s: vars(s)))
    model = model_class(config)
    return model


@torch.no_grad()
def evaluate_reconstruction(model, dm):
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
        required=True,
        type=str,
        help="Encoder configuration",
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
    vae_config, _ = load_config(args.vae_config)

    recon_ect = torch.load(f"results/{vae_config.logger.results_dir}/recon_ect.pt")
    gt_ect = torch.load(f"results/{vae_config.logger.results_dir}/gt_ect.pt")

    plt.imshow(gt_ect[0].squeeze())
    plt.show()

    print(
        f"Loading model from ./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}"
    )
    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)
    encoder_model.eval()

    gt_pc = encoder_model(gt_ect.squeeze().movedim(-1, -2).cuda()).view(-1, 2048, 3)
    print(gt_pc.shape)
    #     recon_pc = encoder_model(recon_ect.squeeze().cuda())
    #
    plot_recon_3d(
        gt_pc.detach().cpu().numpy(),
        gt_pc.detach().cpu().numpy(),  # , filename="recon_newstuff.png"
    )
