import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Conv => LeakyReLU"""

    def __init__(self, in_channels, out_channels=48, batch_norm=False, linear_activation=False):
        super().__init__()
        if not linear_activation:
            if not batch_norm:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                    nn.LeakyReLU(inplace=True, negative_slope=1e-1))
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(inplace=True, negative_slope=1e-1))
        if linear_activation:
            if not batch_norm:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Conv => LeakyReLU (=> Downscaling)"""

    def __init__(self, channels=48):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(channels, channels),
            nn.MaxPool2d(2))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling => DoubleConv"""

    def __init__(self, in_channels, out_channels, mid_channels=-1):
        super().__init__()
        mid_channels = out_channels if (mid_channels == -1) else mid_channels
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(
            Conv(in_channels, mid_channels),
            Conv(mid_channels, out_channels))

    def forward(self, x1, x2):
        x = torch.cat([x2, self.up(x1)], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, linear_activation=True)

    def forward(self, x):
        return self.conv(x)
