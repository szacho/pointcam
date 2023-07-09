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

from pointcam.configs.constants import LOG_PATH
from pointcam.configs.finetune import (
    FinetuneMN40Config,
    FinetuneScanObjBgConfig,
    FinetuneScanObjHardestConfig,
    FinetuneScanObjOnlyConfig,
)
from pointcam.models import PointCAM
from pointcam.models.mlp import MlpClassifier
from pointcam.utils.builders import get_datamodule

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "mps"

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    model = PointCAM.from_exported_checkpoint(args.checkpoint_path, drop_path_rate=0.2)

    match args.config:
        case "mn40":
            finetune_cfg = FinetuneMN40Config()
        case "scanobjbg":
            finetune_cfg = FinetuneScanObjBgConfig()
        case "scanobjonly":
            finetune_cfg = FinetuneScanObjOnlyConfig()
        case "scanobjhardest":
            finetune_cfg = FinetuneScanObjHardestConfig()
        case _:
            raise ValueError(f"Invalid config: {args.config}")

    finetune_cfg = OmegaConf.structured(finetune_cfg)
    print(OmegaConf.to_yaml(finetune_cfg))

    classifier = MlpClassifier(model, finetune_cfg)

    datamodule = get_datamodule(finetune_cfg.data)
    datamodule.setup("fit")

    ckpt_callback = ModelCheckpoint(
        save_weights_only=True,
        save_top_k=1,
        filename="best",
        auto_insert_metric_name=False,
        save_last=True,
        monitor="val/acc",
        mode="max",
    )

    callbacks = [
        ckpt_callback,
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = WandbLogger(
        project="opus", name=finetune_cfg.name, save_dir=str(LOG_PATH), mode="online"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=finetune_cfg.train.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=50,
        precision=finetune_cfg.train.precision,
        logger=logger,
        gradient_clip_val=finetune_cfg.train.clip,
        check_val_every_n_epoch=1,
    )

    trainer.fit(
        classifier,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    print(f"Best accuracy: {classifier.best_acc:.2%}")
