import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from pointcam.configs.constants import LOG_PATH
from pointcam.configs.fewshot import FewShotConfig
from pointcam.datasets import ModelNetFewShotDataModule
from pointcam.models import PointCAM
from pointcam.models.mlp import MlpClassifier

torch.set_float32_matmul_precision("high")


def run_fold(args, fold: int, group_id: str):
    model = PointCAM.from_exported_checkpoint(args.checkpoint_path)

    fewshot_cfg = FewShotConfig(
        name=f"{args.way}way-{args.shot}shot-{fold}",
    )
    fewshot_cfg.model.n_classes = args.way
    fewshot_cfg = OmegaConf.structured(fewshot_cfg)
    print(OmegaConf.to_yaml(fewshot_cfg))

    classifier = MlpClassifier(model, fewshot_cfg)

    datamodule = ModelNetFewShotDataModule(
        way=args.way,
        shot=args.shot,
        fold=fold,
        cfg=model.cfg.data,
    )
    datamodule.setup("fit")

    ckpt_callback = ModelCheckpoint(
        save_weights_only=True,
        save_top_k=1,
        filename="{epoch:03d}-{val/acc:.2f}" + model.cfg.model.name,
        auto_insert_metric_name=False,
        monitor="val/acc",
        mode="max",
    )

    callbacks = [
        ckpt_callback,
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = WandbLogger(
        project="opus",
        name=fewshot_cfg.name,
        save_dir=str(LOG_PATH),
        mode="online",
        group=f"{args.way}way{args.shot}shot-{group_id}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=fewshot_cfg.train.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision=fewshot_cfg.train.precision,
        logger=logger,
        gradient_clip_val=fewshot_cfg.train.clip,
        check_val_every_n_epoch=1,
    )

    trainer.fit(
        classifier,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    wandb.finish()

    return classifier.best_acc.cpu().item()


if __name__ == "__main__":
    pl.seed_everything(11)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("-w", "--way", type=int, default=5)
    parser.add_argument("-s", "--shot", type=int, default=10)
    args = parser.parse_args()

    scores = []
    group_id = wandb.util.generate_id()
    for fold in range(10):
        fold_acc = run_fold(args, fold, group_id)
        scores.append(fold_acc)

    acc_avg = np.mean(scores)
    acc_std = np.std(scores)

    for fold, acc in enumerate(scores):
        print(f"Fold {fold} accuracy: {acc:.2%}")

    print(f"Mean accuracy: {acc_avg:.2%} ± {acc_std:.2%}")
