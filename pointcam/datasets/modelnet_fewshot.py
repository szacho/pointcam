import pickle
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from pointcam.configs.base import DataConfig
from pointcam.configs.constants import DATA_PATH
from pointcam.datasets.base import get_collate_fn, get_transform
from pointcam.utils.crop import PatchFinder, fps
from pointcam.utils.transforms import normalize, to_tensor


class ModelNetFewShot(Dataset):
    def __init__(
        self,
        way: int,
        shot: int,
        fold: int,
        subset: str = "train",
        n_points: int = 1024,
        n_patches=64,
        points_per_patch=32,
        random: bool = True,
        uniform: bool = True,
        transform=None,
    ):
        self.way = way
        self.shot = shot
        self.fold = fold
        self.n_points = n_points
        self.subset = subset
        self.random = random
        self.uniform = uniform
        self.transform = transform

        self.patch_finder = PatchFinder(
            n_patches,
            points_per_patch,
        )
        self.rng = np.random.default_rng()

        self.root = DATA_PATH / "ModelNetFewshot"
        with open(self.root / f"{way}way_{shot}shot/{fold}.pkl", "rb") as f:
            self.data = pickle.load(f)[subset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        xyz, label, _ = self.data[index]
        xyz = xyz[:, 0:3]

        n_points_to_choose = (
            self.n_points if not self.uniform else (self.n_points // 1000) * 1200
        )
        if self.random:
            xyz = self.rng.choice(xyz, size=n_points_to_choose, replace=False)
        else:
            xyz = xyz[:n_points_to_choose]

        normalized_xyz = normalize(to_tensor(xyz))
        xyz = normalized_xyz.unsqueeze(0)

        if self.uniform:
            xyz = fps(xyz, self.n_points, random=self.random)

        if self.transform is not None:
            xyz = self.transform(xyz)

        xyz = self.patch_finder(xyz, random=self.random)

        return {
            "xyz": xyz,
            "raw_xyz": normalized_xyz,
            "label": torch.tensor(label),
        }


class ModelNetFewShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        way: int,
        shot: int,
        fold: int,
        cfg: DataConfig,
    ):
        super().__init__()
        self.way = way
        self.shot = shot
        self.fold = fold

        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.n_points = cfg.n_points
        self.num_workers = cfg.num_workers

        data_dir = cfg.root_path / cfg.dataset.value
        self.paths = [path for path in data_dir.glob("*") if path.is_dir()]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ModelNetFewShot(
            way=self.way,
            shot=self.shot,
            fold=self.fold,
            n_points=self.n_points,
            n_patches=self.cfg.n_patches,
            points_per_patch=self.cfg.points_per_patch,
            uniform=self.cfg.uniform,
            transform=get_transform(self.cfg),
            random=True,
            subset="train",
        )

        self.test_dataset = ModelNetFewShot(
            way=self.way,
            shot=self.shot,
            fold=self.fold,
            n_points=self.n_points,
            n_patches=self.cfg.n_patches,
            points_per_patch=self.cfg.points_per_patch,
            uniform=self.cfg.uniform,
            random=True,
            subset="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.cfg),
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.cfg),
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def flip_random(self, state: bool):
        self.train_dataset.random = state
        self.test_dataset.random = state
