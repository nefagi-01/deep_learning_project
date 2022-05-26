from .helpers import *


class UNet(nn.Module):
    def __init__(self, num_layers=5, num_features=48):
        assert num_layers > 2 and num_layers % 2 == 1
        super().__init__()
        self.num_layers = num_layers + 1
        conv_layers = []
        deconv_layers = []

        conv_layers.append(Conv(3, out_channels=num_features))
        conv_layers.extend([Down(num_features) for i in range(self.num_layers - 1)])
        conv_layers.append(Conv(num_features, out_channels=num_features))

        deconv_layers.append(Up(num_features * 2, num_features * 2))
        deconv_layers.extend([Up(num_features * 3, num_features * 2) for i in range(self.num_layers - 3)])
        deconv_layers.append(Up(num_features * 2 + 3, num_features-16, num_features+16))
        deconv_layers.append(OutConv(num_features-16, 3))

        self.encoder = nn.Sequential(*conv_layers)
        self.decoder = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_feats = [x]
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            if 0 < i < self.num_layers - 1:
                conv_feats.append(x)

        for i in range(self.num_layers):
            if len(conv_feats) > 0:
                x = self.decoder[i](x, conv_feats.pop(-1))
            else:
                x = self.decoder[i](x)

        return x