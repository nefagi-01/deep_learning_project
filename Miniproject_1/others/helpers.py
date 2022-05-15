import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Conv => LeakyReLU (=> Downscaling)"""

    def __init__(self, in_channels, out_channels=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Conv => LeakyReLU (=> Downscaling)"""

    def __init__(self, channels=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling => DoubleConv"""

    def __init__(self, in_channels, out_channels, mid_channels=-1):
        super().__init__()
        mid_channels = out_channels if (mid_channels == -1) else mid_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x1, x2):
        x = torch.cat([x2, self.up(x1)], dim=1)
        return self.conv(x)


class LastUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
