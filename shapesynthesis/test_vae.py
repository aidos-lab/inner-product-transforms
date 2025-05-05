import argparse
import json
import os
from pprint import pprint

import torch
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(model: ModelWrapper, dm):

    result_emd_list = []
    result_cd_list = []

    ect_fn = model.encoder.ect_transform
    for _, batch in enumerate(dm.test_dataloader):
        # First compute the ect
        ect = ect_fn(batch.cuda())
        out_pc, reconstructed_ect = model.reconstruct(ect)

        cd_score, emd_score = EMD_CD(
            out_pc.view(-1, 2048, 3),
            batch,
            batch_size=8,
            reduced=False,
            accelerated_cd=True,
        )
        result_cd_list.append(cd_score)
        result_emd_list.append(emd_score)

    cd = torch.cat(result_cd_list).mean()
    emd = torch.cat(result_emd_list).mean()
    return cd.item(), emd.item()


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

    encoder_model = load_model(
        encoder_config.modelconfig,
        f"./{encoder_config.trainer.save_dir}/{encoder_config.trainer.model_name}",
    ).to(DEVICE)
    encoder_model.model.eval()

    # Set model name for saving results in the results folder.
    model_name = encoder_config.trainer.model_name.split(".")[0]

    dm = load_datamodule(encoder_config.data, dev=args.dev)

    vae_config, _ = load_config(args.vae_config)

    # Check that the configs are equal for the ECT.
    assert vae_config.ectconfig == encoder_config.ectconfig

    vae_model = load_model(
        vae_config.modelconfig,
        f"./{vae_config.trainer.save_dir}/{vae_config.trainer.model_name}",
    ).to(DEVICE)

    # If VAE is provided, overwrite the modelname.
    model_name = vae_config.trainer.model_name.split(".")[0]

    encoder_model.eval()

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        cd_score, emd_score = evaluate_reconstruction(model, dm)
        results.append({"MMD-CD": cd_score, "MMD-EMD": emd_score})

    #####################################################
    ### Saving and printing
    #####################################################

    pprint(results)

    result_suffix = ""
    if args.dev:
        result_suffix = "_dev"

    # Make sure folders exist
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results_dev", exist_ok=True)

    # # Save the results in json format, {config name}.json
    # # Example ./results/encoder_mnist.json
    # with open(
    #     f"./results{result_suffix}/{model_name}/{model_name}{suffix}.json",
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(results, f)
