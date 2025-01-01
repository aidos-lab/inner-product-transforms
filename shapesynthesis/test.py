import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F

import numpy as np

# from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelNoOpWrapper, ModelWrapper

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model: ModelWrapper, dm):
    all_sample = []
    all_ref = []
    all_means = []
    all_std = []
    all_ects = []
    all_recon_ect = []
    for i, batch in enumerate(dm.test_dataloader()):
        out_pc, reconstructed_ect = model.reconstruct(batch.to(DEVICE))
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
        if torch.is_tensor(reconstructed_ect):
            all_ects.append(batch.ect)
            all_recon_ect.append(reconstructed_ect)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    means = torch.cat(all_means, dim=0)
    stdevs = torch.cat(all_std, dim=0)
    if torch.is_tensor(reconstructed_ect):
        reconstructed_ect = torch.cat(all_recon_ect)
        ects = torch.cat(all_ects)
    else:
        reconstructed_ect = None
        ects = None

    if pc_shape[1] == 2:
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

    if pc_shape[1] == 2:
        sample_pcs = sample_pcs[:, :, :2]
        ref_pcs = ref_pcs[:, :, :2]

    results = {
        ("%s" % k): (v if isinstance(v, float) else v.item())
        for k, v in results.items()
    }

    return results, sample_pcs, ref_pcs, reconstructed_ect, ects, means, stdevs


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
    args = parser.parse_args()

    encoder_config, _ = load_config(args.encoder_config)

    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)
    encoder_model.model.eval()

    # Set model name for saving results in the results folder.
    model_name = encoder_config.trainer.model_name.split(".")[0]

    dm = load_datamodule(encoder_config.data)

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

    # Overwrite model if noop is passed
    if args.noop:
        model = ModelNoOpWrapper()
    else:
        model = ModelWrapper(encoder_model, vae_model)

    # for i, batch in enumerate(dm.test_dataloader()):
    #     out_pc = model.reconstruct(batch.to(DEVICE))
    #     break

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, reconstructed_ect, ect, means, stdevs = (
            evaluate_reconstruction(model, dm)
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
    if torch.is_tensor(reconstructed_ect):
        torch.save(
            reconstructed_ect,
            f"./results/{model_name}/reconstructed_ect.pt",
        )
        torch.save(
            ect,
            f"./results/{model_name}/ect.pt",
        )

    pprint(results)
