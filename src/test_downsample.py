import argparse
import json
import os

import torch
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelDownsampleWrapper
from plotting import plot_ect, plot_recon_3d

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model: ModelDownsampleWrapper, dm):
    m = dm.m.view(1, 1, 3).cuda()
    s = dm.s.squeeze().cuda()
    print("STD", s.shape, s)
    print("MEAN", m.shape)

    all_sample = []
    all_ref = []
    all_ects = []
    all_recon_ect = []
    all_sparse_pcs = []
    for idx, pcs in enumerate(dm.val_dataloader):

        ect_gt = model.encoder_downsampler.ect_transform(pcs.cuda())
        out_pc, sparse_pc, reconstructed_ect = model.reconstruct(ect_gt)

        out_pc = out_pc * s + m
        pcs = pcs * dm.s + dm.m
        sparse_pc = sparse_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(pcs)
        all_ects.append(ect_gt)
        all_recon_ect.append(reconstructed_ect)
        all_sparse_pcs.append(sparse_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    sparse_pcs = torch.cat(all_sparse_pcs, dim=0)
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
    return result, sample_pcs, ref_pcs, sparse_pcs, reconstructed_ect, ects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downsampler_config",
        required=True,
        type=str,
        help="Encoder configuration",
    )
    parser.add_argument(
        "--upsampler_config",
        required=True,
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
    encoder_downsampler_config, _ = load_config(args.downsampler_config)

    # Inject dev runs if needed.
    if args.dev:
        encoder_downsampler_config.trainer.save_dir += "_dev"

    print(
        f"Loading model from ./{encoder_downsampler_config.trainer.save_dir}/{encoder_downsampler_config.trainer.model_name}"
    )
    encoder_downsampler = load_model(
        encoder_downsampler_config.modelconfig,
        f"./{encoder_downsampler_config.trainer.save_dir}/{encoder_downsampler_config.trainer.model_name}",
    ).to(DEVICE)
    encoder_downsampler.model.eval()

    # Load the datamodule
    dm = load_datamodule(encoder_downsampler_config.data, dev=args.dev)

    encoder_upsampler_config, _ = load_config(args.upsampler_config)

    if args.dev:
        encoder_upsampler_config.trainer.save_dir += "_dev"

    # Check that the configs are equal for the ECT.
    assert encoder_upsampler_config.ectconfig == encoder_downsampler_config.ectconfig

    encoder_upsampler = load_model(
        encoder_upsampler_config.modelconfig,
        f"./{encoder_upsampler_config.trainer.save_dir}/{encoder_upsampler_config.trainer.model_name}",
    ).to(DEVICE)

    # Define model name
    model_name = encoder_downsampler_config.trainer.model_name.split(".")[0]

    encoder_downsampler.eval()
    encoder_upsampler.eval()

    model = ModelDownsampleWrapper(encoder_downsampler, encoder_upsampler)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, sparse_pc, reconstructed_ect, ect = (
            evaluate_reconstruction(model, dm)
        )

        suffix = ""
        result["model"] = model_name
        result["num_pts"] = encoder_downsampler_config.modelconfig.num_pts
        result["normalized"] = False
        results.append(result)

    #####################################################
    ### Saving and printing
    #####################################################

    result_suffix = ""
    if args.dev:
        result_suffix = "_dev"

    print(results)
    # plot_recon_3d(
    #     sample_pc.cpu().numpy(),
    #     ref_pc.cpu().numpy(),
    #     num_pc=20,
    #     #  filename=f"./results{result_suffix}/{model_name}/reconstruction.png",
    # )
    plot_recon_3d(
        sparse_pc.cpu().numpy(),
        ref_pc.cpu().numpy(),
        num_pc=5,
        point_size=5,
        #  filename=f"./results{result_suffix}/{model_name}/reconstruction.png",
    )

    # Make sure folders exist
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results_dev", exist_ok=True)

    # Save the results in json format, {config name}.json
    # Example ./results/encoder_mnist.json
    with open(
        f"./results{result_suffix}/{model_name}/{model_name}{suffix}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)
    torch.save(
        sample_pc,
        f"./results{result_suffix}/{model_name}/reconstructions{suffix}.pt",
    )
    torch.save(
        ref_pc,
        f"./results{result_suffix}/{model_name}/references{suffix}.pt",
    )
    torch.save(
        sparse_pc,
        f"./results{result_suffix}/{model_name}/sparsereconstruction{suffix}.pt",
    )
    # torch.save(
    #     means,
    #     f"./results{result_suffix}/{model_name}/means.pt",
    # )
    # torch.save(
    #     stdevs,
    #     f"./results{result_suffix}/{model_name}/stdevs.pt",
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
