import argparse
from pprint import pprint

import torch
from loaders import load_config
from metrics.evaluation import EMD_CD
from plotting import plot_recon_3d

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_reconstruction(results_dir: str):

    gt = torch.load(f"{results_dir}/ground_truth.pt")
    pred = torch.load(f"{results_dir}/predictions.pt")

    cd_score, emd_score = EMD_CD(
        pred,
        gt,
        batch_size=100,
        reduced=False,
        accelerated_cd=True,
    )

    cd = cd_score.mean().item()
    emd = emd_score.mean().item()

    return cd, emd


def visualize_results(results_dir):
    gt = torch.load(f"{results_dir}/ground_truth.pt").cpu().numpy()
    pred = torch.load(f"{results_dir}/predictions.pt").cpu().numpy()
    plot_recon_3d(gt, pred, num_pc=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_config",
        required=True,
        type=str,
        help="Encoder configuration",
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()

    ##################################################################
    ### Encoder
    ##################################################################
    config, _ = load_config(args.encoder_config)

    # Modify configs based on dev flag.
    if args.dev:
        config.loggers.tags.append("dev")
        config.trainer.max_epochs = 1
        config.trainer.save_dir += "_dev"
        config.trainer.results_dir += "_dev"

    results_dir = f"{config.trainer.results_dir}/{config.loggers.experiment_name}"

    results = []

    # Evaluate reconstruction
    cd_score, emd_score = evaluate_reconstruction(results_dir)
    results.append({"MMD-CD": cd_score, "MMD-EMD": emd_score})

    #####################################################
    ### Visualize
    #####################################################

    visualize_results(results_dir)

    pprint(results)
