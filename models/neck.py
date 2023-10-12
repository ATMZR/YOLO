import torch
import torch.nn as nn


class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()

        self.output1 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.output2 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.output3 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

        self.merge1 = FPNBlock(out_channels, out_channels)
        self.merge2 = FPNBlock(out_channels, out_channels)

    def forward(self, inputs):
        low_level, mid_level, high_level = inputs
        low_level = self.output1(low_level)
        mid_level = self.output2(mid_level)
        high_level = self.output3(high_level)

        mid_level += nn.functional.interpolate(high_level, scale_factor=2, mode="nearest")
        low_level += nn.functional.interpolate(mid_level, scale_factor=2, mode="nearest")

        mid_level = self.merge1(mid_level)
        low_level = self.merge2(low_level)
        return [low_level, mid_level, high_level]