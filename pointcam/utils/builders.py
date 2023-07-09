import torch

from pointcam.configs import types as t
from pointcam.configs.base import DataConfig, OptimizerConfig
from pointcam.datasets import (
    ModelNetDataModule,
    ScanObjectNNDataModule,
    ShapeNetDataModule,
)
from pointcam.utils.schedulers import ConstantWithWarmupScheduler, TimmCosineLRScheduler


def group_params(model, weight_decay: float):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "token" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0, "name": "no-decay"},
        {"params": decay, "weight_decay": weight_decay, "name": "decay"},
    ]


def group_params_finetune(model, weight_decay: float, encoder_lr: float):
    encoder_decay = []
    encoder_no_decay = []
    mlp = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "student" in name:
            if len(param.shape) == 1 or name.endswith(".bias") or "token" in name:
                encoder_no_decay.append(param)
            else:
                encoder_decay.append(param)
        else:
            mlp.append(param)

    return [
        {
            "params": encoder_no_decay,
            "weight_decay": 0.0,
            "lr": encoder_lr,
            "name": "no-decay",
        },
        {
            "params": encoder_decay,
            "weight_decay": weight_decay,
            "lr": encoder_lr,
            "name": "decay",
        },
        {"params": mlp, "name": "classifier"},
    ]


def get_optimizer(params, total_steps: int, cfg: OptimizerConfig):
    match cfg.name:
        case t.Optimizer.ADAMW:
            optimizer = torch.optim.AdamW(
                params=params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        case t.Optimizer.SGD:
            optimizer = torch.optim.SGD(
                params=params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=0.9,
            )
        case _:
            raise NotImplementedError(f"Optimizer {cfg.name} is not supported.")

    match cfg.scheduler:
        case t.Scheduler.CONSTANT:
            scheduler = ConstantWithWarmupScheduler(
                optimizer=optimizer,
                warmup_steps=int(total_steps * cfg.warmup_ratio),
            )
        case t.Scheduler.ONE_CYCLE:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=cfg.lr,
                total_steps=total_steps,
                pct_start=cfg.warmup_ratio,
            )
        case t.Scheduler.COSINE:
            scheduler = TimmCosineLRScheduler(
                optimizer,
                t_initial=total_steps,
                lr_min=1e-6,
                warmup_lr_init=1e-6,
                warmup_t=int(total_steps * cfg.warmup_ratio),
                cycle_limit=1,
                t_in_epochs=False,
                k_decay=cfg.k_decay,
            )
        case _:
            raise NotImplementedError(f"Scheduler {cfg.scheduler} is not supported.")

    return optimizer, scheduler


def get_datamodule(cfg: DataConfig):
    match cfg.dataset:
        case t.Dataset.MODELNET40 | t.Dataset.MODELNET10 | t.Dataset.MODELNET40FPS:
            datamodule = ModelNetDataModule(cfg=cfg)
        case t.Dataset.SHAPENET | t.Dataset.SHAPENETFPS:
            datamodule = ShapeNetDataModule(cfg=cfg)
        case t.Dataset.SCANOBJECTNN:
            datamodule = ScanObjectNNDataModule(cfg=cfg)
        case _:
            raise NotImplementedError(f"Dataset {cfg.dataset} is not supported.")

    return datamodule
