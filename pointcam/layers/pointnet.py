import torch
from einops import rearrange, reduce, repeat
from torch import nn

from pointcam.utils.crop import PatchedCloud


class Conv1dBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


class TinyPointNet(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = Conv1dBlock(3, 128, 256, kernel_size=1)
        self.conv2 = Conv1dBlock(512, 512, out_dim, kernel_size=1)

    def forward(self, xyz: PatchedCloud):
        """
        Returns:
            feature_global: embeddings of patches, [B, H]
        """
        batch_size = xyz.patches.shape[0]
        n_points = xyz.patches.shape[2]

        patches = rearrange(xyz.patches, "b c n d -> (b c) d n")

        feature = self.conv1(patches)
        feature_global = reduce(feature, "b h n -> b h 1", "max")
        feature_global = repeat(feature_global, "b h 1 -> b h n", n=n_points)

        feature = torch.cat([feature_global, feature], dim=1)
        feature = self.conv2(feature)

        feature_global = reduce(feature, "b h n -> b h", "max")
        return rearrange(feature_global, "(b c) h -> b c h", b=batch_size)
