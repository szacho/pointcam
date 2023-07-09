from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pointcam.configs.base import DataConfig
from pointcam.datasets.base import PointCloudDataset, get_collate_fn, get_transform


class ModelNetDataset(PointCloudDataset):
    def __init__(
        self,
        paths: list[Path],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classes = sorted([path.parent.name for path in paths])
        self.files = self._load_filenames(paths)
        self.labels = self._load_labels(self.files, self.classes)

    @staticmethod
    def _load_filenames(paths: list[Path]):
        files = []
        for path in paths:
            files.extend(path.glob("*.npy"))

        np.random.default_rng(seed=42).shuffle(files)
        return files

    @staticmethod
    def _load_labels(files: list[Path], classes: list[str]):
        labels = [classes.index(file.parents[1].name) for file in files]
        return torch.tensor(labels, dtype=torch.long)


class ModelNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.n_points = cfg.n_points
        self.num_workers = cfg.num_workers

        data_dir = cfg.root_path / cfg.dataset.value
        self.paths = [path for path in data_dir.glob("*") if path.is_dir()]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ModelNetDataset(
            paths=[path / "train" for path in self.paths],
            n_points=self.n_points,
            n_patches=self.cfg.n_patches,
            points_per_patch=self.cfg.points_per_patch,
            uniform=self.cfg.uniform,
            transform=get_transform(self.cfg),
            random=True,
        )

        self.test_dataset = ModelNetDataset(
            paths=[path / "test" for path in self.paths],
            n_points=self.n_points,
            n_patches=self.cfg.n_patches,
            points_per_patch=self.cfg.points_per_patch,
            uniform=self.cfg.uniform,
            random=True,
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
