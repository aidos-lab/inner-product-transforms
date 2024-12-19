import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F

import numpy as np

# from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD, compute_all_metrics
from model_wrapper import ModelWrapper

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_gen(model: ModelWrapper, dm, fast_run: bool):
    all_sample = []
    all_ref = []
    all_sample_ect = []
    all_ects = []
    for i, batch in enumerate(dm.test_dataloader()):
        pc_shape = batch[0].x.shape
        out_pc, sample_ect = model.sample(len(batch), pc_shape[0], pc_shape[1])

        # m, s = data["mean"].float(), data["std"].float()
        if hasattr(batch, "mean") and hasattr(batch, "std"):
            m = torch.tensor(np.stack(batch.mean)).cuda()
            s = torch.tensor(np.stack(batch.std)).cuda()
        else:
            m = torch.zeros(size=(1, 1, pc_shape[-1])).cuda()
            s = torch.ones(size=(1, 1, 1)).cuda()

        out_pc = out_pc * s + m
        te_pc = batch.x.view(-1, pc_shape[0], pc_shape[1]).cuda() * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
        if torch.is_tensor(sample_ect):
            all_ects.append(batch.ect)
            all_sample_ect.append(sample_ect)

        if fast_run and i == 0:
            break

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    if pc_shape[1] == 2:
        sample_pcs = F.pad(
            input=sample_pcs, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0
        )
        ref_pcs = F.pad(input=ref_pcs, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)

    if torch.is_tensor(sample_ect):
        sample_ects = torch.cat(all_sample_ect)
        ects = torch.cat(all_ects)
    else:
        sample_ects = None
        ects = None

    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, 8, accelerated_cd=True)
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    if pc_shape[1] == 2:
        sample_pcs = sample_pcs[:, :, :2]
        ref_pcs = ref_pcs[:, :, :2]
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
        "--fast_run",
        default=False,
        action="store_true",
        help="Only evaluate on the first batch.",
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

    vae_config, _ = load_config(args.vae_config)
    # Check that the configs are equal for the ECT.
    assert vae_config.ectconfig == encoder_config.ectconfig

    vae_model = load_model(
        vae_config.modelconfig,
        f"./{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
    ).to(DEVICE)

    # If VAE is provided, overwrite the modelname.
    model_name = vae_config.trainer.model_name.split(".")[0]

    model = ModelWrapper(encoder_model, vae_model)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, sample_ect, ect = evaluate_gen(
            model, dm, args.fast_run
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
