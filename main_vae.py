import os
import uuid
from loggers import get_wandb_logger
import wandb
from torch_geometric.transforms import Compose
from lightning.pytorch.loggers import WandbLogger
import torch

torch.set_float32_matmul_precision("medium")


from models.base import BaseModel
from models.vae import VanillaVAE

from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_loss_fn


# from loggers import get_wandb_logger
import lightning as L
from omegaconf import OmegaConf


from datasets.mnist import MnistDataModule
from datasets.config import MnistDataModuleConfig
from layers.ect import EctLayer, EctConfig
import torch


CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE_NUM}" if torch.cuda.is_available() else "cpu")

V = torch.vstack(
    [
        torch.sin(torch.linspace(0, 2 * torch.pi, 64, device=DEVICE)),
        torch.cos(torch.linspace(0, 2 * torch.pi, 64, device=DEVICE)),
    ]
)


layer = EctLayer(
    EctConfig(num_thetas=64, bump_steps=64, normalized=True, device=DEVICE), v=V
)


dm = MnistDataModule(MnistDataModuleConfig(root="./data/mnistpointcloud"))


model = VanillaVAE(in_channels=1, latent_dim=64)


litmodel = BaseModel(
    model,
    *get_mse_metrics(),
    accuracies_fn=compute_mse_accuracies,
    loss_fn=compute_mse_loss_fn,
    learning_rate=0.005,
    layer=layer,
)

# metrics = get_mse_metrics()


# litmodel = BaseModel.load_from_checkpoint(
#     "./trained_models/vae.ckpt",
#     model=model,
#     training_accuracy=metrics[0],
#     test_accuracy=metrics[1],
#     validation_accuracy=metrics[2],
#     accuracies_fn=compute_mse_accuracies,
#     loss_fn=compute_mse_loss_fn,
#     learning_rate=0.005,
#     layer=layer,
# )

config = OmegaConf.load("./config.yaml")

logger = get_wandb_logger(config.loggers)

trainer = L.Trainer(
    logger=logger,
    accelerator=config.trainer.accelerator,
    max_epochs=config.trainer.max_epochs,
    log_every_n_steps=config.trainer.log_every_n_steps,
    fast_dev_run=False,
)

trainer.fit(litmodel, dm)
trainer.save_checkpoint("./trained_models/vae.ckpt")
