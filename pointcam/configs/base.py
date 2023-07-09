from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from omegaconf import MISSING

from pointcam.configs import types as t
from pointcam.configs.constants import DATA_PATH


@dataclass
class ModelConfig:
    name: str = MISSING


@dataclass
class TransformerEncoderConfig:
    n_heads: int = 6
    depth: int = 4
    embedding_dim: int = 384
    drop_path_rate: float = 0.1
    drop_rate: float = 0.0

    projection_dim: int = 512
    hidden_dim: int = 1024
    bottleneck_dim: int = 128
    shared_head: bool = True
    ffn_layer: t.FFNLayer = t.FFNLayer.SWIGLU


@dataclass
class MaskNetConfig:
    n_heads: int = 3
    depth: int = 2
    embedding_dim: int = 384
    n_masks: int = 3
    drop_path_rate: float = 0.0
    drop_rate: float = 0.1
    masks_temperature: float = 0.05
    ffn_layer: t.FFNLayer = t.FFNLayer.SWIGLU


@dataclass
class pcBOTModelConfig(ModelConfig):
    name: t.Model = t.Model.pcBOT
    masking_ratios: tuple[float, float] = (0.25, 0.45)
    ibot_weight: float = 1.0
    dino_weight: float = 1.0
    student_temp: float = 0.1
    teacher_temp: tuple[float, float] = (0.04, 0.07)
    cls_center_momentum: float = 0.9
    patch_center_momentum: float = 0.9
    teacher_momentum: tuple[float, float] = (0.994, 1.0)


@dataclass
class PointCAMModelConfig(pcBOTModelConfig):
    name: t.Model = t.Model.PointCAM
    sparsity_weight: float = 0.4
    sparsity_beta: float = 1.0
    diversity_weight: float = 0.0


@dataclass
class MlpClassifierConfig(ModelConfig):
    name: t.Model = t.Model.MLP
    freeze: bool = False
    n_classes: int = 40
    n_layers: int = 3
    smoothing: float = 0.3


@dataclass
class DataConfig:
    root_path: Path = DATA_PATH
    dataset: t.Dataset = t.Dataset.MODELNET40
    scanobjectnn_mode: Optional[t.ScanObjectNNMode] = None
    subset_ratio: float = 1.0

    n_points: int = 1024
    uniform: bool = True
    batch_size: int = 32
    num_workers: int = 24
    pin_memory: bool = False

    n_patches: int = 64
    points_per_patch: int = 32

    n_local_crops: int = 8
    n_global_crops: int = 2
    local_crop_ratios: tuple[float, float] = (0.1, 0.3)
    global_crop_ratios: tuple[float, float] = (0.3, 0.5)

    scale: Optional[tuple[float, float]] = (0.65, 1.5)
    translate: Optional[float] = 0.2
    rotate: Optional[float] = None
    jitter_std: Optional[float] = 0.005


@dataclass
class OptimizerConfig:
    name: t.Optimizer = MISSING
    lr: float = 1e-4
    encoder_lr: Optional[float] = None
    weight_decay: float = 1e-2
    scheduler: t.Scheduler = t.Scheduler.ONE_CYCLE
    warmup_ratio: float = 0.1
    k_decay: float = 1.0


@dataclass
class TrainConfig:
    max_epochs: int = 100
    precision: Union[int, str] = "bf16-mixed"
    clip: Optional[float] = 1.0
    log_config: bool = True
    log_media_every_n_epochs: int = 5
    log_masks_n_batches: int = 2


@dataclass
class ExperimentConfig:
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    train: TrainConfig = MISSING
