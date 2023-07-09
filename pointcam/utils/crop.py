import torch
from einops import rearrange
from pytorch3d.ops import knn_points, sample_farthest_points
from torch import nn


def fps(xyz: torch.Tensor, n_points: int, random: bool = True):
    """
    Args:
        xyz: pointcloud data, [B, N, 3]
        n_points: number of farthest points
        random: sample points randomly
    Returns:
        centroids: sampled points, [B, n_points, 3]
    """
    return sample_farthest_points(xyz, K=n_points, random_start_point=random)[0]


class PatchedCloud:
    def __init__(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
        is_flattened: bool = False,
    ):
        self.patches = patches  # [B, n_crops, n_patches, max_patch_size, 3]
        self.centers = centers  # [B, n_crops, n_patches, 3]

        self.is_flattened = is_flattened
        self.n_crops = patches.shape[1]

    def flatten(self):
        if self.is_flattened:
            return self
        else:
            return self.__class__(
                self.patches.flatten(0, 1),
                self.centers.flatten(0, 1),
                is_flattened=True,
            )

    def to(self, device):
        return self.__class__(
            self.patches.to(device),
            self.centers.to(device),
            is_flattened=self.is_flattened,
        )

    @classmethod
    def stack(cls, clouds: list["PatchedCloud"]):
        if any(cloud.is_flattened for cloud in clouds):
            raise ValueError("Cannot stack flattened PatchedClouds")

        return cls(
            torch.cat([cloud.patches for cloud in clouds], dim=0),
            torch.cat([cloud.centers for cloud in clouds], dim=0),
        )

    @property
    def patches_decentered(self):
        return self.patches + self.centers.unsqueeze(-2)


class PatchFinder(nn.Module):
    def __init__(self, n_patches: int, patch_size: int):
        super().__init__()
        self.n_patches = n_patches
        self.patch_size = patch_size

    def forward(self, xyz, random: bool = True) -> PatchedCloud:
        """
        Args:
            xyz: batch of point clouds, [B, N, 3]
        Returns:
            patches: batch of patches, [B, n_patches, patch_size, 3]
            centers : centers of patches, [B, n_patches, 3]
        """
        batch_size = xyz.shape[0]
        centers = fps(xyz, self.n_patches, random=random)  # B G 3

        idx = knn_points(centers, xyz, K=self.patch_size).idx

        assert idx.size(1) == self.n_patches
        assert idx.size(2) == self.patch_size

        idx = rearrange(idx, "b g n -> b (g n)")
        patches = xyz[torch.arange(batch_size).unsqueeze(1), idx]
        patches = rearrange(patches, "b (g n) c -> b g n c", g=self.n_patches)

        patches = patches - centers.unsqueeze(2)

        return PatchedCloud(
            patches.unsqueeze(0),
            centers.unsqueeze(0),
        )


class CropSampler(nn.Module):
    def __init__(
        self,
        min_ratio: float,
        max_ratio: float,
    ):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(self, xyz: PatchedCloud) -> PatchedCloud:
        n_crops = xyz.patches.shape[1]
        xyz = xyz.flatten()
        batch_size, n_patches = xyz.patches.shape[0], xyz.patches.shape[1]

        crops_centers_idx = torch.multinomial(torch.ones(batch_size, n_patches), 1)
        crops_centers = xyz.centers[
            torch.arange(batch_size).unsqueeze(1), crops_centers_idx
        ]

        crop_length = torch.randint(
            int(self.min_ratio * n_patches),
            int(self.max_ratio * n_patches),
            (1,),
        )

        idx = knn_points(
            crops_centers,
            xyz.centers,
            K=crop_length.item(),
        ).idx

        idx = rearrange(idx, "b c n -> b (c n)")
        cropped_patches = xyz.patches[torch.arange(batch_size).unsqueeze(1), idx]
        cropped_centers = xyz.centers[torch.arange(batch_size).unsqueeze(1), idx]

        cropped_patches = rearrange(
            cropped_patches, "(b c) n g d -> b c n g d", c=n_crops
        )
        cropped_centers = rearrange(cropped_centers, "(b c) n g -> b c n g", c=n_crops)

        return PatchedCloud(
            cropped_patches,
            cropped_centers,
        )
