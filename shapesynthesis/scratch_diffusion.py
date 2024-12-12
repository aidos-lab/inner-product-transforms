from diffusers.models.unets.unet_1d import UNet1DModel
import torch

model = UNet1DModel(
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

input_data = torch.rand(1, 64, 128).cuda()
model(input_data, 0, return_dict=False)
