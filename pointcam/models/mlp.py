import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Accuracy

from pointcam.configs.constants import LOG_PATH
from pointcam.models.pointcam import PointCAM
from pointcam.utils.builders import get_optimizer, group_params_finetune
from pointcam.utils.crop import PatchedCloud


class MlpClassifier(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: PointCAM,
        cfg,
    ):
        super().__init__()
        self.model = pretrained_model
        self.cfg = cfg

        if cfg.model.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        if cfg.model.n_layers == 1:
            self.classifier = nn.Linear(768, cfg.model.n_classes)
        elif cfg.model.n_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, cfg.model.n_classes),
            )
        elif cfg.model.n_layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, cfg.model.n_classes),
            )
        else:
            raise NotImplementedError

        self.train_acc = Accuracy(
            task="multiclass", num_classes=cfg.model.n_classes, average="micro"
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=cfg.model.n_classes, average="micro"
        )

        self.best_acc = 0.0

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.model.smoothing)

    @property
    def name(self):
        mode = "freeze" if self.cfg.model.freeze else "finetune"
        return f"{self.cfg.model.name.value}{self.cfg.model.n_layers}-{mode}"

    def forward(self, xyz: PatchedCloud):
        features = self.model(xyz)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        xyz, label = batch["xyz"], batch["label"]
        logits = self(xyz)
        loss = self.loss_fn(logits, label)

        self.log_dict(
            {"train/loss": loss, "train/acc": self.train_acc(logits, label)},
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        xyz, label = batch["xyz"], batch["label"]
        logits = self(xyz)

        loss = self.loss_fn(logits, label)
        self.val_acc.update(logits, label)

        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        self.log("val/acc", val_acc, prog_bar=True)
        self.val_acc.reset()

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.log("best/acc", self.best_acc, prog_bar=True)

    def configure_optimizers(self):
        params = group_params_finetune(
            self, self.cfg.optimizer.weight_decay, self.cfg.optimizer.encoder_lr
        )

        optimizer, scheduler = get_optimizer(
            params,
            self.trainer.estimated_stepping_batches,
            self.cfg.optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "lr-encoder",
            },
        }

    def on_train_start(self):
        if self.cfg.train.log_config:
            self.logger.log_hyperparams(self.cfg)

            config_path = (
                LOG_PATH
                / self.logger.experiment.project
                / self.logger.experiment.id
                / "config.yaml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, config_path)
