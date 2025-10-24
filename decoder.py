from torch import nn
from segment_anything.modeling.common import LayerNorm2d


class NaiveDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.GELU
        self.map_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            self.activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            self.activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            self.activation(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.map_decoder(x)
