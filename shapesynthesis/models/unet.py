import json
from dataclasses import dataclass
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from datasets.transforms import EctTransformConfig
from loaders import load_model
from models.blocks import DownBlock, MidBlock, UpBlockUnet, get_time_embedding
from models.schedulers.linear_scheduler import (
    LinearNoiseScheduler,
    NoiseSchedulerConfig,
)
from models.vqvae import BaseLightningModel as VAE
from torch import nn
from torchmetrics.regression import MeanSquaredError

from shapesynthesis.datasets.transforms import EctTransform


@dataclass
class ModelConfig:
    module: str
    latent_dim: int
    learning_rate: float
    ectconfig: EctTransformConfig
    beta_period: int
    beta_min: float
    beta_max: float
    down_channels: list[int]
    mid_channels: list[int]
    down_sample: list[bool]
    attn_down: list[bool]
    time_emb_dim: int
    norm_channels: int
    num_heads: int
    conv_out_channels: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int
    module: str
    im_channels: int
    learning_rate: float
    noise_scheduler: NoiseSchedulerConfig
    vae_checkpoint: str


class BaseLightningModel(L.LightningModule):
    """Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()
        self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")

        self.model = Unet(config=self.config)
        self.scheduler = LinearNoiseScheduler(config.noise_scheduler)

        self.vae = VAE.load_from_checkpoint(self.config.vae_checkpoint)

        self.visualization = []

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        return optimizer

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
        pcs_gt = pcs_gt[0]

        im = self.ect_transform(pcs_gt).unsqueeze(1)
        with torch.no_grad():
            im, _ = self.vae.encode(im)

        # Sample random noise
        noise = torch.randn_like(im)

        # Sample timestep
        t = torch.randint(
            0,
            self.config.noise_scheduler.num_timesteps,
            (im.shape[0],),
            device=self.device,
        )

        # Add noise to images according to timestep
        noisy_im = self.scheduler.add_noise(im, noise, t)
        noise_pred = self(noisy_im, t)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss_dict = {
            f"{step}_loss": loss,
        }

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=len(pcs_gt),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.down_channels = config.down_channels
        self.mid_channels = config.mid_channels
        self.t_emb_dim = config.time_emb_dim
        self.down_sample = config.down_sample
        self.num_down_layers = config.num_down_layers
        self.num_mid_layers = config.num_mid_layers
        self.num_up_layers = config.num_up_layers
        self.attns = config.attn_down
        self.norm_channels = config.norm_channels
        self.num_heads = config.num_heads
        self.conv_out_channels = config.conv_out_channels

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(
            config.im_channels,
            self.down_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    attn=self.attns[i],
                    norm_channels=self.norm_channels,
                )
            )

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.t_emb_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(
                    self.down_channels[i] * 2,
                    self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                    self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(
            self.conv_out_channels, config.im_channels, kernel_size=3, padding=1
        )

    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
