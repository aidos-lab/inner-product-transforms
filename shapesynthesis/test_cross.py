import argparse
import json
import os

import torch
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper
from plotting import plot_recon_3d

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
        out_pc, reconstructed_ect = model.reconstruct(ect_gt)

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
        "--cross_config",
        required=False,
        default=None,
        type=str,
        help="Dataset Configuration (Optional)",
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

    cross_config, _ = load_config(args.cross_config)
    dm = load_datamodule(cross_config.data, dev=args.dev)

    if args.dev:
        cross_config.trainer.save_dir += "_dev"

    model_name = cross_config.trainer.model_name.split(".")[0]

    encoder_model.eval()

    model = ModelWrapper(encoder_model)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, reconstructed_ect, ect = evaluate_reconstruction(
            model, dm
        )

        result["model"] = model_name
        result["normalized"] = False
        suffix = ""
        results.append(result)

    #####################################################
    ### Saving and printing
    #####################################################

    result_suffix = ""
    if args.dev:
        result_suffix = "_dev"

    print(result)
    # plot_recon_3d(
    #     sample_pc.cpu().numpy(),
    #     ref_pc.cpu().numpy(),
    #     num_pc=20,
    #     #  filename=f"./results{result_suffix}/{model_name}/reconstruction.png",
    # )

    # torch.save(reconstructed_ect, "recon_ect.pt")
    # torch.save(ect, "gt_ect.pt")

    # # Make sure folders exist
    # os.makedirs("./results", exist_ok=True)
    # os.makedirs("./results_dev", exist_ok=True)
    #
    # # Save the results in json format, {config name}.json
    # # Example ./results/encoder_mnist.json
    # with open(
    #     f"./results{result_suffix}/{model_name}/{model_name}{suffix}.json",
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(results, f)
    # torch.save(
    #     sample_pc,
    #     f"./results{result_suffix}/{model_name}/reconstructions{suffix}.pt",
    # )
    # torch.save(
    #     ref_pc,
    #     f"./results{result_suffix}/{model_name}/references{suffix}.pt",
    # )
    # if torch.is_tensor(reconstructed_ect):
    #     torch.save(
    #         reconstructed_ect,
    #         f"./results{result_suffix}/{model_name}/reconstructed_ect.pt",
    #     )
    #     torch.save(
    #         ect,
    #         f"./results{result_suffix}/{model_name}/ect.pt",
    #     )
