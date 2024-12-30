import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F

import numpy as np

# from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelDownsampleWrapper

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model, dm):
    all_sample = []
    all_ref = []
    all_means = []
    all_std = []
    for i, batch in enumerate(dm.test_dataloader()):
        out_pc = model.reconstruct(batch.to(DEVICE))
        pc_shape = batch[0].x.shape

        # m, s = data["mean"].float(), data["std"].float()
        if hasattr(batch, "mean") and hasattr(batch, "std"):
            m = torch.tensor(np.stack(batch.mean)).cuda()
            s = torch.tensor(np.stack(batch.std)).cuda()
        else:
            m = torch.zeros(size=(1, 1, pc_shape[-1])).cuda()
            s = torch.ones(size=(1, 1, 1)).cuda()

        te_pc = batch.x.view(-1, pc_shape[0], pc_shape[1])
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
        all_means.append(m)
        all_std.append(s)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    means = torch.cat(all_means, dim=0)
    stdevs = torch.cat(all_std, dim=0)

    if pc_shape[1] == 2:
        sample_pcs = F.pad(
            input=sample_pcs, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0
        )
        ref_pcs = F.pad(input=ref_pcs, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)

    print(sample_pcs.shape)
    print(ref_pcs.shape)
    results = EMD_CD(
        sample_pcs,
        ref_pcs,
        batch_size=8,
        reduced=True,
        accelerated_cd=True,
    )

    if pc_shape[1] == 2:
        sample_pcs = sample_pcs[:, :, :2]
        ref_pcs = ref_pcs[:, :, :2]

    results = {
        ("%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in results.items()
    }

    return results, sample_pcs, ref_pcs, means, stdevs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_downsample_config",
        required=True,
        type=str,
        help="Encoder downsampler configuration",
    )
    parser.add_argument(
        "--encoder_upsample_config",
        required=True,
        type=str,
        help="Encoder upsampler (should be default encoder model) configuration",
    )
    parser.add_argument(
        "--num_reruns",
        default=1,
        type=int,
        help="Number of reruns for the standard deviation.",
    )

    args = parser.parse_args()

    encoder_downsample_config, _ = load_config(args.encoder_downsample_config)
    encoder_upsample_config, _ = load_config(args.encoder_upsample_config)

    encoder_downsample_model = load_model(
        encoder_downsample_config.modelconfig,
        f"./{encoder_downsample_config.trainer.save_dir}/{encoder_downsample_config.trainer.model_name}",
    ).to(DEVICE)

    encoder_upsample_model = load_model(
        encoder_upsample_config.modelconfig,
        f"./{encoder_upsample_config.trainer.save_dir}/{encoder_upsample_config.trainer.model_name}",
    ).to(DEVICE)

    # Set model name for saving results in the results folder.
    model_name = encoder_downsample_config.trainer.model_name.split(".")[0]

    # Load data
    dm = load_datamodule(encoder_downsample_config.data)

    # Load the datamodule
    # NOTE: Loads the datamodule from the encoder and does not check for
    # equality of the VAE data configs.

    model = ModelDownsampleWrapper(encoder_downsample_model, encoder_upsample_model)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, means, stdevs = evaluate_reconstruction(model, dm)
        result["normalized"] = False
        result["model"] = model_name

        suffix = ""

        results.append(result)

    print("SAVING RESULTS", model_name)
    # Save the results in json format, {config name}.json
    # Example ./results/encoder_mnist.json
    with open(
        f"./results/{model_name}/{model_name}{suffix}.json",
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
        means,
        f"./results/{model_name}/means.pt",
    )
    torch.save(
        stdevs,
        f"./results/{model_name}/stdevs.pt",
    )

    pprint(results)
