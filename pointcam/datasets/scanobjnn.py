from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from pointcam.configs.base import DataConfig
from pointcam.configs.types import ScanObjectNNMode
from pointcam.datasets.base import get_collate_fn, get_transform
from pointcam.utils.crop import PatchFinder
from pointcam.utils.transforms import normalize


class ScanObjNNDataset(Dataset):
    def __init__(
        self,
        n_patches: int = 64,
        points_per_patch: int = 32,
        root_path: Path = None,
        mode: ScanObjectNNMode = ScanObjectNNMode.OBJ_BG,
        subset: str = "train",
        transform=None,
        random: bool = True,
    ):
        self.patch_finder = PatchFinder(
            n_patches,
            points_per_patch,
        )
        self.transform = transform
        self.random = random

        self.data_path = self.get_data_path(root_path, mode, subset)
        self.points, self.labels = self.load_data(self.data_path)

    @staticmethod
    def get_data_path(root_path: Path, mode: ScanObjectNNMode, subset: str):
        if subset == "train":
            subset_prefix = "training"
        else:
            subset_prefix = subset

        match mode:
            case ScanObjectNNMode.OBJ_BG:
                data_path = (
                    root_path / "main_split" / f"{subset_prefix}_objectdataset.h5"
                )
            case ScanObjectNNMode.PB_T50_RS:
                data_path = (
                    root_path
                    / "main_split"
                    / f"{subset_prefix}_objectdataset_augmentedrot_scale75.h5"
                )
            case ScanObjectNNMode.OBJ_ONLY:
                data_path = (
                    root_path / "main_split_nobg" / f"{subset_prefix}_objectdataset.h5"
                )
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        return data_path

    def load_data(self, path: Path):
        h5 = h5py.File(path, "r")
        points = np.array(h5["data"]).astype(np.float32)
        labels = np.array(h5["label"]).astype(int)
        h5.close()

        return torch.tensor(points, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.long
        )

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx: int):
        xyz = self.points[idx]

        if self.random:
            points_idx = torch.randperm(xyz.shape[0])
            xyz = xyz[points_idx]

        normalized_xyz = normalize(xyz)
        xyz = normalized_xyz.unsqueeze(0)

        if self.transform is not None:
            xyz = self.transform(xyz)

        xyz = self.patch_finder(xyz, random=self.random)

        return {
            "xyz": xyz,
            "raw_xyz": normalized_xyz,
            "label": self.labels[idx],
        }


class ScanObjectNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        self.data_path = cfg.root_path / cfg.dataset.value

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ScanObjNNDataset(
            root_path=self.data_path,
            n_patches=self.cfg.n_patches,
            points_per_patch=self.cfg.points_per_patch,
            subset="train",
            mode=self.cfg.scanobjectnn_mode,
            transform=get_transform(self.cfg),
            random=True,
        )
        self.test_dataset = ScanObjNNDataset(
            root_path=self.data_path,
            n_patches=self.cfg.n_patches,
            points_per_patch=self.cfg.points_per_patch,
            subset="test",
            mode=self.cfg.scanobjectnn_mode,
            random=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.cfg),
            drop_last=True,
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
