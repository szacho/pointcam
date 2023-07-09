import argparse

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from pointcam.configs import PointCAMConfig, pcBOTConfig
from pointcam.configs.constants import LOG_PATH
from pointcam.models import PointCAM, pcBOT
from pointcam.utils.builders import get_datamodule

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="pointcam")
    args = parser.parse_args()

    if args.model == "pointcam":
        cfg = OmegaConf.structured(PointCAMConfig)
        model = PointCAM(cfg)
    elif args.model == "pcbot":
        cfg = OmegaConf.structured(pcBOTConfig)
        model = pcBOT(cfg)
    else:
        raise ValueError(f"Unknown model: {args.model}, choose from pointcam, pcbot.")

    print(OmegaConf.to_yaml(cfg))
    datamodule = get_datamodule(cfg.data)

    ckpt_callback = ModelCheckpoint(
        save_weights_only=False,
        save_top_k=-1,
        every_n_epochs=cfg.train.log_media_every_n_epochs,
        filename="{epoch:03d}-" + cfg.model.name,
        auto_insert_metric_name=False,
    )

    callbacks = [
        ckpt_callback,
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = WandbLogger(
        project="pointcam",
        name=model.name,
        save_dir=str(LOG_PATH),
        mode="online",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=50,
        precision=cfg.train.precision,
        logger=logger,
    )

    trainer.fit(model, datamodule)
