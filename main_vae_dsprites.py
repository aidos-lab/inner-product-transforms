from datasets.dsprites import DspritesDataModule, DspritesDataModuleConfig
from directions import generate_2d_directions
from loggers import get_wandb_logger
import torch

torch.set_float32_matmul_precision("medium")


from models.vae_mnist import VanillaVAE, BaseModel

from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_kld_loss_fn

import lightning as L
from omegaconf import OmegaConf

from layers.ect import EctLayer, EctConfig
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

config = OmegaConf.load("./configs/config_vae_dsprites.yaml")

layer = EctLayer(
    EctConfig(
        num_thetas=config.layer.ect_size,
        bump_steps=config.layer.ect_size,
        normalized=True,
        device=DEVICE,
    ),
    v=generate_2d_directions(config.layer.ect_size, DEVICE),
)


dm = DspritesDataModule(
    DspritesDataModuleConfig(root="./data/dsprites", batch_size=256)
)

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
