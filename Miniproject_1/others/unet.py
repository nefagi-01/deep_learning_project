from .helpers import *


class UNet(nn.Module):
    def __init__(self, num_layers=6, num_features=48):
        assert num_layers > 3
        super().__init__()
        self.num_layers = num_layers
        conv_layers = []
        deconv_layers = []

        conv_layers.append(Conv(3))
        conv_layers.extend([Down(num_features) for i in range(num_layers - 1)])
        conv_layers.append(Conv(num_features))


        deconv_layers.append(Up(num_features * 2, num_features * 2))
        deconv_layers.extend([Up(num_features * 3, num_features * 2) for i in range(num_layers - 3)])
        deconv_layers.append(Up(num_features * 2 + 3, 32, 64))
        deconv_layers.append(OutConv(32, 3))

        self.encoder = nn.Sequential(*conv_layers)
        self.decoder = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)
        # # INITIALIZE ENCODER
        # self.enc = Conv(3)
        # self.down1 = Down(48)
        # self.down2 = Down(48)
        # self.down3 = Down(48)
        # self.down4 = Down(48)
        # self.down5 = Down(48)
        # self.down6 = Conv(48)
        # # INITIALIZE DECODER
        # self.up1 = Up(96, 96)
        # self.up2 = Up(144, 96)
        # self.up3 = Up(144, 96)
        # self.up4 = Up(144, 96)
        # self.up5 = Up(99, 32, 64)
        # self.outc = OutConv(32, 3)

    def forward(self, x):
        conv_feats = [x]
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            if 0 < i < self.num_layers - 1:
                conv_feats.append(x)

        for i in range(self.num_layers):
            if len(conv_feats)>0:
                x = self.decoder[i](x, conv_feats.pop(-1))
            else:
                x = self.decoder[i](x)

        return x

        # # ENCODER
        # enc_conv0 = self.down0(x)
        # pool1 = self.down1(enc_conv0)
        # pool2 = self.down2(pool1)
        # pool3 = self.down3(pool2)
        # pool4 = self.down4(pool3)
        # pool5 = self.down5(pool4)
        # enc_conv6 = self.down6(pool5)
        #
        # # DECODER
        # dec_conv5 = self.up1(enc_conv6, pool4)
        # dec_conv4 = self.up2(dec_conv5, pool3)
        # dec_conv3 = self.up3(dec_conv4, pool2)
        # dec_conv2 = self.up4(dec_conv3, pool1)
        # dec_conv1 = self.up5(dec_conv2, x)
        # return self.outc(dec_conv1)
