import argparse
import json
from pprint import pprint

import torch
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper

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
    for _, batch in enumerate(dm.test_dataloader()):
        out_pc, reconstructed_ect = model.reconstruct(batch.to(DEVICE))
        pc_shape = batch[0].x.shape
        m, s = batch.mean, batch.std

        te_pc = batch.x.view(-1, pc_shape[0], pc_shape[1])

        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
        all_means.append(m)
        all_std.append(s)
        all_ects.append(batch.ect)
        all_recon_ect.append(reconstructed_ect)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    means = torch.cat(all_means, dim=0)
    stdevs = torch.cat(all_std, dim=0)
    reconstructed_ect = torch.cat(all_recon_ect)
    ects = torch.cat(all_ects)

    results = EMD_CD(
        sample_pcs,
        ref_pcs,
        batch_size=8,
        reduced=True,
        accelerated_cd=True,
    )

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

        # If VAE is provided, overwrite the modelname.
        model_name = vae_config.trainer.model_name.split(".")[0]
    else:
        vae_model = None

    encoder_model.eval()

    model = ModelWrapper(encoder_model, vae_model)

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc, reconstructed_ect, ect, means, stdevs = (
            evaluate_reconstruction(model, dm)
        )

        suffix = ""
        results.append(result)

    #####################################################
    ### Saving and printing
    #####################################################

    pprint(results)

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
