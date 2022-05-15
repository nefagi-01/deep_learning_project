from others.helpers import *


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # INITIALIZE ENCODER
        self.down0 = Conv(3)
        self.down1 = Down(48)
        self.down2 = Down(48)
        self.down3 = Down(48)
        self.down4 = Down(48)
        self.down5 = Down(48)
        self.down6 = Conv(48)
        # INITIALIZE DECODER
        self.up1 = Up(96, 96)
        self.up2 = Up(144, 96)
        self.up3 = Up(144, 96)
        self.up4 = Up(144, 96)
        self.up5 = Up(99, 32, 64)
        self.outc = LastUp(32, 3)

    def forward(self, x):
        # ENCODER
        enc_conv0 = self.down0(x)
        pool1 = self.down1(enc_conv0)
        pool2 = self.down2(pool1)
        pool3 = self.down3(pool2)
        pool4 = self.down4(pool3)
        pool5 = self.down5(pool4)
        enc_conv6 = self.down6(pool5)

        # DECODER
        dec_conv5 = self.up1(enc_conv6, pool4)
        dec_conv4 = self.up2(dec_conv5, pool3)
        dec_conv3 = self.up3(dec_conv4, pool2)
        dec_conv2 = self.up4(dec_conv3, pool1)
        dec_conv1 = self.up5(dec_conv2, x)
        return self.outc(dec_conv1)
