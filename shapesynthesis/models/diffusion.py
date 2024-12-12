from typing import TypeAlias, Literal
import torch
import lightning as L
import torch.nn.functional as F
from diffusers.models.unets.unet_1d import UNet1DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

Tensor: TypeAlias = torch.Tensor


class BaseLightningModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = UNet1DModel(
            # sample_rate=1,
            sample_size=128,  # the target image resolution
            in_channels=64,  # the number of input channels, 3 for RGB images
            out_channels=64,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                128,
                128,
                # 256,
                # 256,
                256,
                128,
            ),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock1D",  # a regular ResNet downsampling block
                "DownBlock1D",
                # "DownBlock1D",
                # "DownBlock1D",
                "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock1D",
            ),
            up_block_types=(
                "UpBlock1D",  # a regular ResNet upsampling block
                "AttnUpBlock1D",  # a ResNet upsampling block with spatial self-attention
                # "UpBlock1D",
                # "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
            ),
        ).cuda()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def forward(self, noisy_images, timesteps):  # pylint: disable=arguments-differ
        return self.model(noisy_images, timesteps, return_dict=False)[0]

    def general_step(self, batch, _, step: Literal["train", "test", "validation"]):
        clean_images = batch.ect.movedim(-1, -2)

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)

        # TODO: NOT IMPLEMENTED.
        # accelerator.clip_grad_norm_(model.parameters(), 1.0)

        self.log_dict(
            {
                f"{step}_loss": loss,
            },
            prog_bar=True,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")
