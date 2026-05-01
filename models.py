import torch
import torch.nn as nn

class GlobalFeatureEnhancement(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GlobalFeatureEnhancement, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.avg_pool(x).view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Reweight
        return x * y

class LocalFeatureEnhancement(nn.Module):
    def __init__(self, channels):
        super(LocalFeatureEnhancement, self).__init__()

        self.conv = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        attn = self.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * attn

