import torch
import lightning as L
from omegaconf import OmegaConf

from datasets.mnist import MnistDataModule, MnistDataModuleConfig
from models.ectencoder_mnist import BaseModel
from layers.ect import EctLayer, EctConfig

from directions import generate_2d_directions
from loggers import get_wandb_logger


# Settings
torch.set_float32_matmul_precision("medium")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

config = OmegaConf.load("./configs/config_encoder_mnist.yaml")


dm = MnistDataModule(MnistDataModuleConfig(root="./data/mnistpointcloud"))

layer = EctLayer(
    EctConfig(
        num_thetas=config.layer.ect_size,
        bump_steps=config.layer.ect_size,
        normalized=True,
        device=DEVICE,
    ),
    v=generate_2d_directions(config.layer.ect_size, DEVICE),
)


encodermodel = BaseModel(
    layer=layer,
    ect_size=config.layer.ect_size,
    hidden_size=config.model.hidden_size,
    num_pts=config.model.num_pts,
    num_dims=config.model.num_dims,
    learning_rate=config.model.learning_rate,
)


logger = get_wandb_logger(config.loggers)

trainer = L.Trainer(
    logger=logger,
    accelerator=config.trainer.accelerator,
    max_epochs=config.trainer.max_epochs,
    log_every_n_steps=config.trainer.log_every_n_steps,
    fast_dev_run=False,
)


trainer.fit(encodermodel, dm)
trainer.save_checkpoint(f"./trained_models/{config.model.save_name}")
