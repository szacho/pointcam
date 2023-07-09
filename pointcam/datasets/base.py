import numpy as np
import torch
from torch.utils.data import Dataset

from pointcam.configs.base import DataConfig
from pointcam.utils.crop import CropSampler, PatchedCloud, PatchFinder, fps
from pointcam.utils.transforms import (
    Jitter,
    Rotate,
    Scale,
    Translate,
    normalize,
    to_tensor,
)


def get_transform(cfg: DataConfig):
    transforms = []
    if cfg.scale is not None:
        transforms.append(Scale(cfg.scale))
    if cfg.rotate is not None:
        transforms.append(Rotate(cfg.rotate))
    if cfg.jitter_std is not None:
        transforms.append(Jitter(cfg.jitter_std))
    if cfg.translate is not None:
        transforms.append(Translate(cfg.translate))

    transform = transforms[0]
    for t in transforms[1:]:
        transform = transform | t

    return transform


class PointCloudDataset(Dataset):
    def __init__(
        self,
        n_points: int = 2048,
        n_patches: int = 64,
        points_per_patch: int = 24,
        n_crops_total: int = 1,
        uniform: bool = False,
        transform=None,
        random: bool = True,
    ):
        self.n_points = n_points
        self.uniform = uniform
        self.transform = transform
        self.random = random

        self.classes = None
        self.files = None
        self.labels = None
        self.rng = np.random.default_rng()
        self.patch_finder = PatchFinder(
            n_patches,
            points_per_patch,
        )

        self.n_crops_total = max(n_crops_total, 1)

    def resample(self, ratio: float):
        n_samples = int(len(self.files) * ratio)
        indices = np.arange(len(self.files))
        sampled_indices = self.rng.choice(indices, size=n_samples, replace=False)

        self.files = [
            file_ for idx, file_ in enumerate(self.files) if idx in sampled_indices
        ]
        self.labels = [
            label for idx, label in enumerate(self.labels) if idx in sampled_indices
        ]

    def __len__(self):
        return len(self.files)

    def repeat_point_cloud(self, xyz):
        xyz = xyz.unsqueeze(0)
        xyz = xyz.repeat(self.n_crops_total, 1, 1)
        return xyz

    def __getitem__(self, idx: int):
        path = self.files[idx]
        xyz = np.load(path)
        normalized_xyz = normalize(to_tensor(xyz))

        xyz = self.repeat_point_cloud(normalized_xyz)

        n_points_to_choose = (
            self.n_points if not self.uniform else (self.n_points // 1000) * 1200
        )
        if self.random:
            indices = torch.multinomial(
                torch.ones(self.n_crops_total, xyz.shape[1]),
                n_points_to_choose,
                replacement=False,
            )
            xyz = xyz[torch.arange(self.n_crops_total).unsqueeze(-1), indices]
        else:
            xyz = xyz[:, :n_points_to_choose]

        if self.uniform:
            xyz = fps(xyz, self.n_points, random=self.random)

        if self.transform is not None:
            xyz = self.transform(xyz)

        xyz = self.patch_finder(xyz, random=self.random)

        return {
            "xyz": xyz,
            "raw_xyz": normalized_xyz,
            "label": self.labels[idx],
        }


def get_collate_fn(cfg: DataConfig, do_crops: bool = False):
    if do_crops:
        local_crop = CropSampler(*cfg.local_crop_ratios)
        global_crop = CropSampler(*cfg.global_crop_ratios)

    def collate_fn(data):
        xyz = PatchedCloud.stack([d["xyz"] for d in data])
        raw_xyz = torch.stack([d["raw_xyz"] for d in data])
        labels = torch.stack([d["label"] for d in data])

        batch = {
            "xyz": xyz,
            "raw_xyz": raw_xyz,
            "label": labels,
        }

        if "seg_label" in data[0]:
            seg_labels = torch.stack([d["seg_label"] for d in data])
            batch["seg_label"] = seg_labels

        if do_crops:
            local_xyz = PatchedCloud(
                xyz.patches[:, : cfg.n_local_crops],
                xyz.centers[:, : cfg.n_local_crops],
            )
            global_xyz = PatchedCloud(
                xyz.patches[:, cfg.n_local_crops :],
                xyz.centers[:, cfg.n_local_crops :],
            )
            batch["local_xyz"] = local_crop(local_xyz)
            batch["global_xyz"] = global_crop(global_xyz)

        return batch

    return collate_fn
