from torch import nn
import math


class REDNet(nn.Module):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNet, self).__init__()
        self.num_layers = num_layers
        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))

        deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0:
                conv_feats.append(x)

        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + self.num_layers) % 2 == 0:
                conv_feat = conv_feats.pop(-1)
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x
