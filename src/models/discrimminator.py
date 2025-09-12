import pydantic
import torch
import torch.nn as nn


class DiscrimminatorConfig(pydantic.BaseModel):
    im_channels: int
    lr: float


class Discrimminator(nn.Module):
    r"""
    PatchGAN Discriminator.
    Rather than taking IMG_CHANNELSxIMG_HxIMG_W all the way to
    1 scalar value , we instead predict grid of values.
    Where each grid is prediction of how likely
    the discriminator thinks that the image patch corresponding
    to the grid cell is real
    """

    def __init__(self, config: DiscrimminatorConfig):
        super().__init__()
        conv_channels = [64, 128, 256]
        kernels = [4, 4, 4, 4]
        strides = [2, 2, 2, 1]
        paddings = [1, 1, 1, 1]

        self.im_channels = config.im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        layers_dim[i],
                        layers_dim[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=paddings[i],
                        bias=False if i != 0 else True,
                    ),
                    (
                        nn.BatchNorm2d(layers_dim[i + 1])
                        if i != len(layers_dim) - 2 and i != 0
                        else nn.Identity()
                    ),
                    activation if i != len(layers_dim) - 2 else nn.Identity(),
                )
                for i in range(len(layers_dim) - 1)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
