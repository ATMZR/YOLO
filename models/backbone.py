import torch
import torch.nn as nn


class DarknetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DarknetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x


class DarknetResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DarknetResidualBlock, self).__init__()
        self.conv1 = DarknetBlock(in_channels, middle_channels, 1, 1, 0)
        self.conv2 = DarknetBlock(middle_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        # Low level
        self.conv1 = DarknetBlock(3, 32, 3, 1, 1)
        self.conv2 = DarknetBlock(32, 64, 3, 2, 1)
        self.residual_block1 = self.__make_residual_block(64, 32, 64, 1)
        self.conv3 = DarknetBlock(64, 128, 3, 2, 1)
        self.residual_block2 = self.__make_residual_block(128, 64, 128, 2)
        self.conv4 = DarknetBlock(128, 256, 3, 2, 1)
        self.residual_block3 = self.__make_residual_block(256, 128, 256, 8)
        # Mid level
        self.conv5 = DarknetBlock(256, 512, 3, 2, 1)
        self.residual_block4 = self.__make_residual_block(512, 256, 512, 8)
        # High level
        self.conv6 = DarknetBlock(512, 1024, 3, 2, 1)
        self.residual_block5 = self.__make_residual_block(1024, 512, 1024, 8)

    @staticmethod
    def __make_residual_block(in_channels, middle_channels, out_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(DarknetResidualBlock(in_channels, middle_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        x = self.conv3(x)
        x = self.residual_block2(x)
        x = self.conv4(x)
        low_level = self.residual_block3(x)
        x = self.conv5(low_level)
        mid_level = self.residual_block4(x)
        x = self.conv6(mid_level)
        high_level = self.residual_block5(x)
        return low_level, mid_level, high_level
