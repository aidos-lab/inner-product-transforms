import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F

# from model_wrapper import ModelWrapper
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_render_reconstruction(folder):

    sample_pcs = torch.load(f"./results/{folder}/reconstructions.pt")
    ref_pcs = torch.load(f"./results/{folder}/references.pt")
    means = torch.load(f"./results/{folder}/means.pt")
    stdevs = torch.load(f"./results/{folder}/stdevs.pt")
    pc_shape = ref_pcs.shape

    sample_pcs = sample_pcs * stdevs + means
    ref_pcs = ref_pcs * stdevs + means

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

    return results, sample_pcs, ref_pcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        required=True,
        type=str,
        help="Folder where the reconstructions are stored.",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="Normalize data before passing it to the model.",
    )
    parser.add_argument(
        "--num_reruns",
        default=1,
        type=int,
        help="Number of reruns for the standard deviation.",
    )
    args = parser.parse_args()

    model_name = args.folder

    results = []
    for _ in range(args.num_reruns):
        # Evaluate reconstruction
        result, sample_pc, ref_pc = evaluate_render_reconstruction(model_name)
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
    pprint(results)
