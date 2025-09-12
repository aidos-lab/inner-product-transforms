import pydantic
import torch
import torch.nn as nn

from models.blocks import DownBlock, MidBlock
from src.layers.ect import EctConfig


class ModelConfig(pydantic.BaseModel):
    module: str
    learning_rate: float
    ectconfig: EctConfig
    ectlossconfig: EctConfig
    num_pts: int
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


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.down_channels = self.config.down_channels
        self.mid_channels = self.config.mid_channels
        self.down_sample = self.config.down_sample
        self.num_down_layers = self.config.num_down_layers
        self.num_mid_layers = self.config.num_mid_layers
        self.num_up_layers = self.config.num_up_layers

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = self.config.attn_down

        # Latent Dimension
        self.num_pts = self.config.num_pts
        self.norm_channels = self.config.norm_channels
        self.num_heads = self.config.num_heads

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(
            config.im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1)
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
                    normtype="batch",
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
                    normtype="batch",
                )
            )

        # self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_norm_out = nn.BatchNorm2d(self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1],
            self.num_pts,
            kernel_size=3,
            padding=1,
        )

        # # Latent Dimension is 2*Latent because we are predicting mean & variance
        # self.pre_quant_conv = nn.Conv2d(
        #     self.num_pts,
        #     self.num_pts,
        #     kernel_size=1,
        # )
        self.final = nn.Sequential(
            nn.Linear(16**2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        ####################################################

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        # out = self.pre_quant_conv(out)
        out = self.final(out.flatten(start_dim=2))
        return out

    def forward(self, x):
        encoder_output = self.encode(x)
        return encoder_output
