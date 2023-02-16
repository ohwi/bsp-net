from functools import partial

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride,
        act=partial(nn.LeakyReLU, negative_slope=0.01), padding_mode="same",
    ):
        super().__init__()

        if padding_mode == "same":
            padding = kernel_size // stride // 2
        elif padding_mode == "valid":
            padding = 0
        else:
            raise NotImplementedError

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = act()

    def forward(self, x):
        x = self.conv0(x)
        x = self.act(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=partial(nn.LeakyReLU, negative_slope=0.01)):
        super().__init__()

        self.fc0 = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
        )
        self.act = act()

    def forward(self, x):
        x = self.fc0(x)
        x = self.act(x)
        return x


# TODO; this model is based on image size 64.
#   Need to generalize.
class Encoder(nn.Module):
    def __init__(self, in_channels, num_planes, base_channels, num_conv_blocks=5, num_fc_blocks=3):
        super().__init__()

        self.num_planes = num_planes
        self.num_conv_blocks = num_conv_blocks
        self.num_fc_blocks = num_fc_blocks

        # convolutional shared part; from layer 0-3
        out_channels = base_channels
        for i in range(num_conv_blocks - 1):
            layer = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
            )
            setattr(self, f"conv{i}", layer)

            in_channels = out_channels
            out_channels = in_channels * 2

        # last layer has different stride, padding_mode
        layer = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=1,
            padding_mode="valid",
        )
        setattr(self, f"conv{num_conv_blocks - 1}", layer)

        # linear shared part; from layer 0-2
        for i in range(num_fc_blocks):
            layer = LinearBlock(
                in_channels=in_channels,
                out_channels=out_channels,
            )
            setattr(self, f"fc{i}", layer)

            in_channels = out_channels
            out_channels = in_channels * 2

        # output linear layer
        self.weights = LinearBlock(
            in_channels=in_channels,
            out_channels=num_planes * 2,    # a and b of ax+by+c = 0
            act=nn.Identity,
        )

        self.bias = LinearBlock(
            in_channels=in_channels,
            out_channels=num_planes,        # c of ax+by+c = 0
            act=nn.Identity,
        )

    def forward(self, x):
        # forward pass of conv layers
        for i in range(self.num_conv_blocks):
            layer = getattr(self, f"conv{i}")
            x = layer(x)

        # flatten
        x = x.flatten(1)

        # forward pass of fc layers
        for i in range(self.num_fc_blocks):
            layer = getattr(self, f"fc{i}")
            x = layer(x)

        weights = self.weights(x)
        bias = self.bias(x)

        num_batch = weights.size(0)
        weights = weights.reshape(num_batch, 2, self.num_planes)
        bias = bias.reshape(num_batch, 1, self.num_planes)

        return weights, bias

