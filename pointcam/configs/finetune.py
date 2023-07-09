from dataclasses import dataclass

from omegaconf import MISSING

from pointcam.configs import types as t
from pointcam.configs.base import (
    DataConfig,
    MlpClassifierConfig,
    OptimizerConfig,
    TrainConfig,
)


@dataclass
class FinetuneMN40Config:
    name: str = "MLP3-Finetune-ModelNet40"
    model: MlpClassifierConfig = MlpClassifierConfig(
        freeze=False,
        n_layers=3,
        smoothing=0.2,
    )
    data: DataConfig = DataConfig(
        dataset=t.Dataset.MODELNET40FPS,
        n_points=1024,
        batch_size=32,
        points_per_patch=32,
        n_patches=64,
        n_local_crops=0,
        n_global_crops=0,
        scale=(0.6, 1.4),
        translate=None,
        rotate=None,
        jitter_std=None,
        uniform=False,
    )
    optimizer: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        lr=1e-3,
        encoder_lr=1e-3,
        weight_decay=0.05,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.05,
        k_decay=1.0,
    )
    train: TrainConfig = TrainConfig(
        max_epochs=150,
        log_media_every_n_epochs=0,
        precision="bf16-mixed",
        clip=10.0,
    )


@dataclass
class FinetuneScanObj:
    name: str = MISSING
    model: MlpClassifierConfig = MlpClassifierConfig(
        freeze=False,
        n_layers=3,
        n_classes=15,
        smoothing=0.3,
    )
    optimizer: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        lr=5e-4,
        encoder_lr=5e-4,
        weight_decay=0.05,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.05,
        k_decay=1.0,
    )
    train: TrainConfig = TrainConfig(
        max_epochs=300,
        log_media_every_n_epochs=0,
        precision="bf16-mixed",
        clip=10.0,
    )


scan_object_data_params = dict(
    n_points=2048,
    batch_size=32,
    points_per_patch=32,
    n_patches=128,
    n_local_crops=0,
    n_global_crops=0,
    scale=(0.6, 1.4),
    translate=0.2,
    rotate=1,
    jitter_std=0.005,
)


@dataclass
class FinetuneScanObjBgConfig(FinetuneScanObj):
    name: str = "MLP3-Finetune-ScObjNN-OBJ-BG"
    data: DataConfig = DataConfig(
        dataset=t.Dataset.SCANOBJECTNN,
        scanobjectnn_mode=t.ScanObjectNNMode.OBJ_BG,
        **scan_object_data_params,
    )


@dataclass
class FinetuneScanObjOnlyConfig(FinetuneScanObj):
    name: str = "MLP3-Finetune-ScObjNN-OBJ-ONLY"
    data: DataConfig = DataConfig(
        dataset=t.Dataset.SCANOBJECTNN,
        scanobjectnn_mode=t.ScanObjectNNMode.OBJ_ONLY,
        **scan_object_data_params,
    )


@dataclass
class FinetuneScanObjHardestConfig(FinetuneScanObj):
    name: str = "MLP3-Finetune-ScObjNN-HARDEST"
    data: DataConfig = DataConfig(
        dataset=t.Dataset.SCANOBJECTNN,
        scanobjectnn_mode=t.ScanObjectNNMode.PB_T50_RS,
        **scan_object_data_params,
    )
