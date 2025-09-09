from dataclasses import dataclass
from typing import Literal

import lightning as L
import pydantic
import torch
import torch.nn as nn
import torchvision
from torch import nn
from torch.optim import Adam
from torchmetrics.regression import MeanSquaredError
from torchvision.utils import make_grid

from layers.ect import EctConfig
from models.blocks import DownBlock, MidBlock, UpBlock
from models.lpips import LPIPS


class ModelConfig(pydantic.BaseModel):
    lr: float
    module: str
    ectconfig: EctConfig
    z_channels: int
    codebook_size: int
    down_channels: list[int]
    mid_channels: list[int]
    down_sample: list[bool]
    attn_down: list[bool]
    norm_channels: int
    im_channels: int
    num_heads: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int
    disc_start: int
    disc_weight: float
    codebook_weight: float
    commitment_beta: float
    perceptual_weight: float
    kl_weight: float


class Model(nn.Module):
    """VQVAE implementation."""

    def __init__(self, config):
        super().__init__()
        self.down_channels = config.down_channels
        self.mid_channels = config.mid_channels
        self.down_sample = config.down_sample
        self.num_down_layers = config.num_down_layers
        self.num_mid_layers = config.num_mid_layers
        self.num_up_layers = config.num_up_layers

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = config.attn_down

        # Latent Dimension
        self.z_channels = config.z_channels
        self.codebook_size = config.codebook_size
        self.norm_channels = config.norm_channels
        self.num_heads = config.num_heads

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(
            config.im_channels,
            self.down_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    t_emb_dim=None,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    attn=self.attns[i],
                    norm_channels=self.norm_channels,
                )
            )

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1], self.z_channels, kernel_size=3, padding=1
        )

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        ####################################################

        ##################### Decoder ######################

        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(
            self.z_channels, self.z_channels, kernel_size=1
        )
        self.decoder_conv_in = nn.Conv2d(
            self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1)
        )

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i - 1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(
                UpBlock(
                    self.down_channels[i],
                    self.down_channels[i - 1],
                    t_emb_dim=None,
                    up_sample=self.down_sample[i - 1],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    attn=self.attns[i - 1],
                    norm_channels=self.norm_channels,
                )
            )

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(
            self.down_channels[0], config.im_channels, kernel_size=3, padding=1
        )

    def quantize(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(
            self.embedding.weight, 0, min_encoding_indices.view(-1)
        )

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commmitment_loss,
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape(
            (-1, quant_out.size(-2), quant_out.size(-1))
        )
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)

        internal_loss = (
            self.config.codebook_weight * quant_losses["codebook_loss"]
        ) + (self.config.commitment_beta * quant_losses["commitment_loss"])
        return out, internal_loss, quant_losses


class BaseLightningModel(L.LightningModule):
    """Base model for VAE models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__()
        self.automatic_optimization = False
        self.training_accuracy = MeanSquaredError()
        self.validation_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()
        self.ect_transform = EctTransform(config=config.ectconfig, device="cuda")

        # # Metrics
        # self.train_fid = FrechetInceptionDistance(
        #     feature=64, normalize=True, input_img_size=(1, 128, 128)
        # )
        # self.val_fid = FrechetInceptionDistance(
        #     feature=64, normalize=True, input_img_size=(1, 128, 128)
        # )
        # self.sample_fid = FrechetInceptionDistance(
        #     feature=64, normalize=True, input_img_size=(1, 128, 128)
        # )

        self.model = VQVAE(config=self.config)

        self.discriminator = Discriminator(self.config)
        self.lpips_model = LPIPS().eval()

        self.recon_criterion = torch.nn.MSELoss()
        self.disc_criterion = torch.nn.MSELoss()
        self.visualization = []

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer_d = Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )
        optimizer_g = Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )

        return [optimizer_g, optimizer_d], []

    def forward(self, batch):  # pylint: disable=arguments-differ
        x = self.model(batch)
        return x

    def general_step(self, pcs_gt, _, step: Literal["train", "test", "validation"]):
        # pcs_gt = pcs_gt[0]

        optimizer_g, optimizer_d = self.optimizers()

        ect_gt = self.ect_transform(pcs_gt).unsqueeze(1)

        # Start adding the discrimminator after 1k steps.
        disc_scale_loss = 0
        # TODO: Fix this.
        if self.global_step > 880:
            disc_scale_loss = 1

        # Fetch autoencoders output(reconstructions)
        output, _, quantize_losses = self(ect_gt)

        ######### Optimize Generator ##########
        self.toggle_optimizer(optimizer_g)

        # L2 Loss
        recon_loss = self.recon_criterion(output, ect_gt)
        g_loss = (
            recon_loss
            + (self.config.codebook_weight * quantize_losses["codebook_loss"])
            + (self.config.commitment_beta * quantize_losses["commitment_loss"])
        )

        # Adversarial loss only if disc_step_start steps passed
        disc_fake_pred = self.discriminator(output)
        disc_fake_loss = self.disc_criterion(
            disc_fake_pred,
            torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
        )
        g_loss += disc_scale_loss * self.config.disc_weight * disc_fake_loss

        lpips_loss = torch.mean(self.lpips_model(output, ect_gt))
        g_loss += self.config.perceptual_weight * lpips_loss

        # TODO: How to do this?
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        ########################################
        ######### Optimize Discriminator #######
        ########################################

        self.toggle_optimizer(optimizer_d)
        fake = output
        disc_fake_pred = self.discriminator(fake.detach())
        disc_real_pred = self.discriminator(ect_gt)
        disc_fake_loss = self.disc_criterion(
            disc_fake_pred,
            torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
        )
        disc_real_loss = self.disc_criterion(
            disc_real_pred,
            torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
        )
        disc_loss = self.config.disc_weight * (disc_fake_loss + disc_real_loss) / 2

        self.manual_backward(disc_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        loss_dict = {
            f"{step}_g_loss": g_loss,
            f"{step}_d_loss": disc_loss,
        }

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=len(pcs_gt),
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    @torch.no_grad()
    def validation_step(self, pcs_gt, batch_idx):
        # pcs_gt = pcs_gt[0]
        ect_gt = self.ect_transform(pcs_gt).unsqueeze(1)

        # Fetch autoencoders output(reconstructions)
        output, _, quantize_losses = self(ect_gt)

        # L2 Loss
        recon_loss = self.recon_criterion(output, ect_gt)
        g_loss = (
            recon_loss
            + (self.config.codebook_weight * quantize_losses["codebook_loss"])
            + (self.config.commitment_beta * quantize_losses["commitment_loss"])
        )

        # Adversarial loss only if disc_step_start steps passed
        disc_fake_pred = self.discriminator(output)
        disc_fake_loss = self.disc_criterion(
            disc_fake_pred,
            torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
        )
        g_loss += self.config.disc_weight * disc_fake_loss

        lpips_loss = torch.mean(self.lpips_model(output, ect_gt))
        g_loss += self.config.perceptual_weight * lpips_loss

        fake = output
        disc_fake_pred = self.discriminator(fake.detach())
        disc_real_pred = self.discriminator(ect_gt)
        disc_fake_loss = self.disc_criterion(
            disc_fake_pred,
            torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
        )
        disc_real_loss = self.disc_criterion(
            disc_real_pred,
            torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
        )
        disc_loss = self.config.disc_weight * (disc_fake_loss + disc_real_loss) / 2

        loss_dict = {
            "validation_g_loss": g_loss,
            "validation_d_loss": disc_loss,
        }

        sample_size = 8
        save_recon = (
            1 + torch.clamp(output[:sample_size], -1.0, 1.0).detach().cpu()
        ) / 2
        save_gt = ((ect_gt[:sample_size] + 1) / 2).detach().cpu()

        grid = make_grid(torch.cat([save_gt, save_recon], dim=0), nrow=sample_size)
        # img = torchvision.transforms.ToPILImage()(grid)
        # img.save(f"results/ect_recon_{self.global_step}.png")
        # img.close()
        if batch_idx == 0:
            self.logger.log_image(
                "generated_images",
                [
                    grid,
                ],
                0,
            )

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=len(pcs_gt),
            on_step=False,
            on_epoch=True,
        )

    def training_step(self, batch, batch_idx):
        self.model.train()
        return self.general_step(batch, batch_idx, "train")
