import torch
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")

from datasets.topological import TopolocigalDataModule, TopologicalDataModuleConfig

from models.vae_mnist import VanillaVAE, BaseModel
from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_kld_loss_fn

import lightning as L
from directions import generate_3d_directions
from loggers import get_wandb_logger

from layers.ect import EctLayer, EctConfig


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

config = OmegaConf.load("./configs/config_vae_topological.yaml")


layer = EctLayer(
    EctConfig(
        num_thetas=config.layer.ect_size,
        bump_steps=config.layer.ect_size,
        normalized=True,
        device=DEVICE,
    ),
    v=generate_3d_directions(config.layer.ect_size, DEVICE),
)

dm = TopolocigalDataModule(TopologicalDataModuleConfig())


model = VanillaVAE(
    in_channels=config.model.in_channels,
    latent_dim=config.model.latent_dim,
    img_size=config.layer.ect_size,
)


litmodel = BaseModel(
    model,
    *get_mse_metrics(),
    accuracies_fn=compute_mse_accuracies,
    loss_fn=compute_mse_kld_loss_fn,
    learning_rate=config.litmodel.learning_rate,
    layer=layer,
)

logger = get_wandb_logger(config.loggers)

trainer = L.Trainer(
    logger=logger,
    accelerator=config.trainer.accelerator,
    max_epochs=config.trainer.max_epochs,
    log_every_n_steps=config.trainer.log_every_n_steps,
    fast_dev_run=False,
)

trainer.fit(litmodel, dm)
trainer.save_checkpoint(f"./trained_models/{config.model.save_name}")
