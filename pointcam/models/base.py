import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from timm.models.layers import trunc_normal_
from torch import nn
from umap import UMAP

from pointcam.configs.base import ExperimentConfig
from pointcam.configs.constants import LOG_PATH
from pointcam.utils.visualize import plot_embeddings


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BaseModel(pl.LightningModule):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg

    @property
    def name(self):
        return f"{self.cfg.model.name.value}-{self.cfg.data.dataset.value}"

    @torch.no_grad()
    def extract_features(self, dataloader):
        features = []
        labels = []

        for batch in dataloader:
            xyz, batch_labels = batch["xyz"], batch["label"]
            batch_features = self(xyz.to(self.device))

            features.append(batch_features)
            labels.append(batch_labels)

        features = torch.cat(features, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        return features, labels

    def should_log_media(self, epoch: int):
        if self.cfg.train.log_media_every_n_epochs <= 0 or self.global_step == 0:
            return False

        if epoch == 0:
            return True

        return (epoch + 1) % self.cfg.train.log_media_every_n_epochs == 0

    def log_umap(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        classes: np.ndarray,
    ):
        labels = classes[y_test]  # int -> str

        reducer = UMAP(n_components=2, random_state=42, metric="cosine")
        reducer = reducer.fit(X_train)
        reduced_embeddings = reducer.transform(X_test)
        reduced_embeddings = {
            category: reduced_embeddings[labels == category]
            for category in np.unique(labels)
        }

        fig = plot_embeddings(
            *reduced_embeddings.values(),
            labels=list(reduced_embeddings.keys()),
        )

        wandb.log({"UMAP": fig})
        umap_path = (
            LOG_PATH
            / self.logger.experiment.project
            / self.logger.experiment.id
            / "UMAP"
        )
        umap_path.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(umap_path / f"{self.current_epoch}.html"))

    def eval_linear(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        clf = LinearSVC(random_state=42, C=5e-3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

        wandb.log(
            {
                "test/accuracy": acc,
                "test/f1-score": f1,
                "test/precision": precision,
                "test/recall": recall,
            }
        )

    def on_train_epoch_end(self):
        self.eval()

        if self.should_log_media(self.current_epoch):
            classes = np.array(self.trainer.datamodule.test_linear_dataset.classes)

            X_train, y_train = self.extract_features(
                self.trainer.datamodule.train_linear_dataloader()
            )
            X_test, y_test = self.extract_features(
                self.trainer.datamodule.test_linear_dataloader()
            )

            self.log_umap(X_train, X_test, y_test, classes)
            self.eval_linear(X_train, y_train, X_test, y_test)

            del X_train, y_train, X_test, y_test

        self.train()

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
