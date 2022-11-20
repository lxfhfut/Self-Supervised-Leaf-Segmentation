from torch import nn


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BackBone(nn.Module):
    def __init__(self, blocks, layers, channels):
        super(BackBone, self).__init__()
        self.num_blocks = len(blocks)
        assert self.num_blocks == len(layers)
        assert self.num_blocks == len(channels) - 1

        self.conv0 = LightConv3x3(3, channels[0])
        self.convs = nn.ModuleList()
        for i in range(self.num_blocks):
            self.convs.append(self.__class__._make_layer(blocks[i], layers[i], channels[i], channels[i+1]))

        self.m = nn.Softmax2d()

    def forward(self, x):
        x = self.conv0(x)
        for i in range(self.num_blocks):
            x = self.convs[i](x)

        out = self.m(x)
        return out

    @staticmethod
    def _make_layer(block, layer, in_channels, out_channels):
        layers = []

        layers.append(block(in_channels, out_channels))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
