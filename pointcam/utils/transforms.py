import numpy as np
import torch
from einops import einsum, rearrange

from pointcam.utils.crop import PatchedCloud


def normalize(xyz: torch.Tensor) -> torch.Tensor:
    centroid = torch.mean(xyz, dim=0)
    xyz -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(xyz**2, dim=-1)))
    xyz /= furthest_distance

    return xyz


def to_tensor(xyz: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(xyz).float()


class BaseBatchTransform:
    def __init__(self, val):
        self.val = val
        self.chain = []

    @staticmethod
    def rand(a: float, b: float, size: tuple[int], device: torch.device):
        return torch.rand(*size, device=device) * (b - a) + a

    def _transform(self, xyz: PatchedCloud) -> PatchedCloud:
        raise NotImplementedError

    def __call__(self, xyz: PatchedCloud) -> PatchedCloud:
        xyz = self._transform(xyz)

        if not self.chain:
            return xyz
        else:
            for other in self.chain:
                xyz = other._transform(xyz)

            return xyz

    def __or__(self, other):
        self.chain.append(other)
        return self


class Jitter(BaseBatchTransform):
    def __init__(self, val: float = 0.01, clip: float = 0.05):
        super().__init__(val)
        self.clip = clip

    def _transform(self, xyz: torch.Tensor) -> torch.Tensor:
        noise = (torch.randn_like(xyz) * self.val).clip(-self.clip, self.clip)
        xyz = xyz + noise.to(xyz.device)

        return xyz


class Scale(BaseBatchTransform):
    def _transform(self, xyz: torch.Tensor) -> torch.Tensor:
        batch_size = xyz.shape[0]

        s_xyz = self.rand(*self.val, (batch_size, 3), xyz.device)
        return xyz * rearrange(s_xyz, "b d -> b 1 d")


class Translate(BaseBatchTransform):
    def _transform(self, xyz: torch.Tensor) -> torch.Tensor:
        batch_size = xyz.shape[0]

        d_xyz = self.rand(-self.val, self.val, (batch_size, 3), xyz.device)
        return xyz + rearrange(d_xyz, "b d -> b 1 d")


class Rotate(BaseBatchTransform):
    def yaw_matrix(self, yaw):
        cos_val = torch.cos(yaw)
        sin_val = torch.sin(yaw)

        zero = torch.zeros_like(cos_val)
        one = torch.ones_like(cos_val)

        return torch.stack(
            [
                torch.stack([cos_val, -sin_val, zero], dim=1),
                torch.stack([sin_val, cos_val, zero], dim=1),
                torch.stack([zero, zero, one], dim=1),
            ],
            dim=1,
        )

    def pitch_matrix(self, pitch):
        cos_val = torch.cos(pitch)
        sin_val = torch.sin(pitch)

        zero = torch.zeros_like(cos_val)
        one = torch.ones_like(cos_val)

        return torch.stack(
            [
                torch.stack([cos_val, zero, sin_val], dim=1),
                torch.stack([zero, one, zero], dim=1),
                torch.stack([-sin_val, zero, cos_val], dim=1),
            ],
            dim=1,
        )

    def roll_matrix(self, roll):
        cos_val = torch.cos(roll)
        sin_val = torch.sin(roll)

        zero = torch.zeros_like(cos_val)
        one = torch.ones_like(cos_val)

        return torch.stack(
            [
                torch.stack([one, zero, zero], dim=1),
                torch.stack([zero, cos_val, -sin_val], dim=1),
                torch.stack([zero, sin_val, cos_val], dim=1),
            ],
            dim=1,
        )

    def _transform(self, xyz: torch.Tensor) -> torch.Tensor:
        batch_size = xyz.shape[0]

        theta = self.rand(
            -self.val * np.pi,
            self.val * np.pi,
            (batch_size, 1),
            xyz.device,
        )

        rot = self.pitch_matrix(theta[:, 0])
        return einsum(xyz, rot, "b i j, b k j -> b i k")
