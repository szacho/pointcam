from dataclasses import dataclass

from pointcam.configs import types as t
from pointcam.configs.base import (
    DataConfig,
    ExperimentConfig,
    MaskNetConfig,
    OptimizerConfig,
    PointCAMModelConfig,
    TrainConfig,
    TransformerEncoderConfig,
    pcBOTModelConfig,
)


@dataclass
class pcBOTConfig(ExperimentConfig):
    model: pcBOTModelConfig = pcBOTModelConfig(
        masking_ratios=(0.1, 0.45),
        ibot_weight=1.0,
        dino_weight=1.0,
        teacher_momentum=(0.994, 1.0),
    )
    encoder: TransformerEncoderConfig = TransformerEncoderConfig(
        n_heads=6,
        depth=12,
        embedding_dim=384,
        projection_dim=512,
        hidden_dim=1024,
        bottleneck_dim=256,
        drop_path_rate=0.1,
        drop_rate=0.0,
        shared_head=False,
        ffn_layer=t.FFNLayer.SWIGLU,
    )
    data: DataConfig = DataConfig(
        dataset=t.Dataset.SHAPENETFPS,
        n_points=1024,
        uniform=True,
        batch_size=128,
        num_workers=24,
        pin_memory=False,
        points_per_patch=32,
        n_local_crops=8,
        n_global_crops=2,
        local_crop_ratios=(0.1, 0.3),
        global_crop_ratios=(0.3, 0.5),
        scale=(0.6, 1.4),
        translate=0.2,
        rotate=None,
        jitter_std=0.005,
    )
    optimizer: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        lr=1e-4,
        weight_decay=5e-2,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.05,
        k_decay=1.5,
    )
    train: TrainConfig = TrainConfig(
        max_epochs=100,
        log_media_every_n_epochs=5,
        log_masks_n_batches=1,
        precision="bf16-mixed",
        clip=1.0,
    )


@dataclass
class PointCAMConfig(pcBOTConfig):
    model: PointCAMModelConfig = PointCAMModelConfig(
        sparsity_weight=0.2,
        diversity_weight=0.3,
        ibot_weight=1.0,
        dino_weight=1.0,
        teacher_momentum=(0.994, 1.0),
    )
    masknet: MaskNetConfig = MaskNetConfig(
        n_heads=4,
        depth=3,
        embedding_dim=384,
        n_masks=4,
        drop_path_rate=0.1,
        drop_rate=0.0,
        masks_temperature=1.0,
        ffn_layer=t.FFNLayer.SWIGLU,
    )
    optimizer_mask: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        lr=3e-5,
        weight_decay=1e-2,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.05,
        k_decay=1.5,
    )
