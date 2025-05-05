import argparse
import os
from pprint import pprint

import lightning as L
import torch
from loaders import load_config, load_datamodule, load_model
from metrics.evaluation import EMD_CD
from model_wrapper import ModelWrapper

# Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class SaveTestOutput(L.Callback):
    def __init__(self, results_dir: str, encoder_ckpt: str | None = None):
        super().__init__()
        self.results_dir = results_dir
        self.ground_truth_batches = []
        self.predicted_batches = []

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: tuple,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        self.ground_truth_batches.append(outputs[0])
        self.predicted_batches.append(outputs[1])
        return super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_test_end(self, trainer, pl_module):
        print("RUN EPOCH END")
        gt = torch.cat(self.ground_truth_batches)
        pred = torch.cat(self.predicted_batches)
        torch.save(gt, f"{self.results_dir}/ground_truth.pt")
        torch.save(pred, f"{self.results_dir}/predictions.pt")
        return super().on_test_end(trainer, pl_module)


def generate_model_output():
    pass


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
