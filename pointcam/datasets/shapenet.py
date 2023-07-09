from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pointcam.configs.base import DataConfig
from pointcam.configs.constants import SYNSETID_TO_LABEL
from pointcam.datasets.base import PointCloudDataset, get_collate_fn, get_transform
from pointcam.datasets.modelnet import ModelNetDataset


class ShapeNetDataset(PointCloudDataset):
    def __init__(
        self, data_dir: Path, subset: str, subset_ratio: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.classes = sorted(SYNSETID_TO_LABEL.values())
        self.files = self._load_files(data_dir, subset)
        self.labels = self._load_labels(self.files, self.classes)

        if subset_ratio < 1.0:
            self.resample(subset_ratio)

    def _load_files(self, data_dir: Path, subset: str):
        split_df = pd.read_csv(data_dir / "all.csv")

        test_split_df = split_df[split_df["split"] == "test"]
        test_model_ids = test_split_df["modelId"].values.tolist()
        test_model_ids = {id_: True for id_ in test_model_ids}

        files = list(data_dir.glob("**/*.npy"))
        if subset == "train":
            files = [
                file_
                for file_ in files
                if not test_model_ids.get(file_.parents[1].name, False)
            ]
        else:
            files = [
                file_
                for file_ in files
                if test_model_ids.get(file_.parents[1].name, False)
            ]

        return files

    @staticmethod
    def _load_labels(files: list[Path], classes: list[str]):
        labels = []
        for file in files:
            synset_id = file.parents[2].name
            label = SYNSETID_TO_LABEL[synset_id]
            labels.append(classes.index(label))

        return torch.tensor(labels, dtype=torch.long)


class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.n_points = cfg.n_points
        self.num_workers = cfg.num_workers

        self.data_dir = cfg.root_path / cfg.dataset.value
        modelnet_data_dir = cfg.root_path / "ModelNet40FPS"
        self.modelnet_paths = [
            path for path in modelnet_data_dir.glob("*") if path.is_dir()
        ]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ShapeNetDataset(
                data_dir=self.data_dir,
                subset="train",
                subset_ratio=self.cfg.subset_ratio,
                n_points=self.n_points,
                n_patches=self.cfg.n_patches,
                points_per_patch=self.cfg.points_per_patch,
                uniform=self.cfg.uniform,
                n_crops_total=self.cfg.n_global_crops + self.cfg.n_local_crops,
                transform=get_transform(self.cfg),
                random=True,
            )
        if stage in ("fit", "test") or stage is None:
            self.train_linear_dataset = ModelNetDataset(
                paths=[path / "train" for path in self.modelnet_paths],
                n_points=self.n_points,
                n_patches=self.cfg.n_patches,
                points_per_patch=self.cfg.points_per_patch,
                uniform=False,
                random=False,
            )

            self.test_linear_dataset = ModelNetDataset(
                paths=[path / "test" for path in self.modelnet_paths],
                n_points=self.n_points,
                n_patches=self.cfg.n_patches,
                points_per_patch=self.cfg.points_per_patch,
                uniform=False,
                random=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.cfg, do_crops=True),
            drop_last=True,
        )

    def train_linear_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.train_linear_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.cfg),
        )

    def test_linear_dataloader(self):
        return DataLoader(
            self.test_linear_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.cfg),
        )

    def flip_random(self, state: bool):
        self.train_linear_dataset.random = state
        self.test_linear_dataset.random = state
