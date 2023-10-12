import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(Head, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.sequent = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), kernel_size=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        low_level, mid_level, high_level = inputs
        low_level = self.sequent(low_level)
        mid_level = self.sequent(mid_level)
        high_level = self.sequent(high_level)

        return [low_level, mid_level, high_level]
