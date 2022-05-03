from others.helpers import *


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def initializeEncoder():
            self.inc = DoubleConv(3, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)

        def initializeDecoder():
            self.up1 = Up(1024, 512)
            self.up2 = Up(512, 256)
            self.up3 = Up(256, 128)
            self.up4 = Up(128, 64)
            self.outc = OutConv(64, 3)

        initializeEncoder()
        initializeDecoder()

    def forward(self, x):
        # ENCODER
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        return self.outc(x9)
