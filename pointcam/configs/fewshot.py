from dataclasses import dataclass

from pointcam.configs import types as t
from pointcam.configs.base import (
    DataConfig,
    MlpClassifierConfig,
    OptimizerConfig,
    TrainConfig,
)


@dataclass
class FewShotConfig:
    name: str = ""  # injected later
    model: MlpClassifierConfig = MlpClassifierConfig(
        freeze=False,
        n_layers=3,
        smoothing=0.3,
    )
    data: DataConfig = DataConfig(
        dataset=t.Dataset.MODELNET40,
        n_points=1024,
        batch_size=32,
        points_per_patch=32,
        n_patches=64,
        n_local_crops=0,
        n_global_crops=0,
        scale=(0.6, 1.4),
        translate=0.2,
        rotate=None,
        jitter_std=0.005,
        uniform=True,
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
        max_epochs=150,
        log_media_every_n_epochs=0,
        precision="bf16-mixed",
        clip=10.0,
    )
