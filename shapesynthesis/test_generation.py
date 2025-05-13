import argparse
import json
from pprint import pprint
from typing import TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

# from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD, compute_all_metrics
from model_wrapper import ModelWrapper
from plotting import plot_ect, plot_recon_3d

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
Tensor: TypeAlias = torch.Tensor


def get_means(batch) -> tuple[Tensor, Tensor]:
    # m, s = data["mean"].float(), data["std"].float()
    if hasattr(batch, "mean") and hasattr(batch, "std"):
        if not isinstance(batch.mean, torch.Tensor):
            m = torch.tensor(np.stack(batch.mean)).cuda()
            s = torch.tensor(np.stack(batch.std)).cuda()
        else:
            m = batch.mean.unsqueeze(1).cuda()
            s = batch.std.unsqueeze(1).cuda()
    else:
        m = torch.zeros(size=(1, 1, batch.x.shape[-1])).cuda()
        s = torch.ones(size=(1, 1, 1)).cuda()
    return m, s


@torch.no_grad()
def evaluate_gen(model: ModelWrapper, dm, dev: bool):
    m = dm.m.view(1, 1, 3).cuda()
    s = dm.s.squeeze().cuda()
    all_sample = []
    all_ref = []
    all_sample_ect = []
    all_ects = []
    for i, pcs_gt in enumerate(dm.val_dataloader):
        ect_gt = model.encoder.ect_transform(pcs_gt.cuda())
        out_pc, sample_ect = model.sample(len(pcs_gt))

        # Scale and translate
        out_pc = out_pc * s + m
        pcs_gt = pcs_gt.cuda() * s + m

        all_sample.append(out_pc)
        all_ref.append(pcs_gt)
        all_ects.append(ect_gt)
        all_sample_ect.append(sample_ect)

        if dev and i == 0:
            break

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    sample_ects = torch.cat(all_sample_ect)
    ects = torch.cat(all_ects)

    print("ECT", ects.min(), ects.max(), ects.shape)
    print("ECT SAMPLE", sample_ects.min(), sample_ects.max(), sample_ects.shape)

    result_suffix = ""
    if args.dev:
        result_suffix = "_dev"

    if args.dev:
        plot_ect(
            ects,
            sample_ects,
            num_ects=2,
            filename=f"./ect.png",
        )
        plot_recon_3d(
            7 * ref_pcs.cpu().numpy(),
            7 * sample_pcs.cpu().numpy(),
            num_pc=20,
            filename=f"./samples.png",
        )
    else:
        plot_ect(
            ects,
            sample_ects,
            num_ects=2,
            # filename=f"./results{result_suffix}/{model_name}/ect.png",
        )
        plot_recon_3d(
            ref_pcs.cpu().numpy(),
            sample_pcs.cpu().numpy(),
            num_pc=20,
            # filename=f"./results{result_suffix}/{model_name}/reconstruction.png",
        )

    # Compute metric
    results = compute_all_metrics(sample_pcs, ref_pcs, 100, accelerated_cd=True)
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }

    pprint(results)

    return results, sample_pcs, ref_pcs, sample_ects, ects


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
        "--normalize",
        default=False,
        action="store_true",
        help="Normalize data before passing it to the model.",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Only evaluate on the first batch.",
    )
    args = parser.parse_args()

    encoder_config, _ = load_config(args.encoder_config)
    if args.dev:
        encoder_config.trainer.save_dir += "_dev"

    print(
        "LOADING:",
        f"{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
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

    vae_config, _ = load_config(args.vae_config)
    print(vae_config)
    # Check that the configs are equal for the ECT.
    assert vae_config.ectconfig == encoder_config.ectconfig

    if args.dev:
        vae_config.trainer.save_dir += "_dev"

    print("LOADING:", f"{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}")
    vae_model = load_model(
        vae_config.modelconfig,
        f"{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
    ).to(DEVICE)

    model_name = vae_config.trainer.model_name.split(".")[0]

    model = ModelWrapper(encoder_model, vae_model)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, sample_ect, ect = evaluate_gen(model, dm, args.dev)
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
        f"./results/{model_name}/{model_name}_gen{suffix}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)
    torch.save(
        sample_pc,
        f"./results/{model_name}/samples{suffix}.pt",
    )
    torch.save(
        ref_pc,
        f"./results/{model_name}/references{suffix}.pt",
    )

    if torch.is_tensor(sample_ect):
        torch.save(
            sample_ect,
            f"./results/{model_name}/sample_ect.pt",
        )
        torch.save(
            ect,
            f"./results/{model_name}/ect.pt",
        )

    pprint(results)
