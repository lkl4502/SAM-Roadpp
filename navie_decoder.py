import torch

from torch import nn
from segment_anything.modeling.common import LayerNorm2d


class NaiveDecoder(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=2,
        channel_list=[128, 64, 32],
        activation=nn.GELU,
    ):
        super().__init__()
        layers = []
        prev_channels = in_channels

        for idx, ch in enumerate(channel_list):
            layers.append(
                nn.ConvTranspose2d(prev_channels, ch, kernel_size=2, stride=2)
            )
            if idx == 0:
                layers.append(LayerNorm2d(128))
            layers.append(activation())
            prev_channels = ch

        layers.append(
            nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)
