from typing import Optional

import matplotlib
import numpy as np
import torch
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from PIL import ImageColor

from pointcam.configs import PointCAMConfig
from pointcam.configs.constants import COLORS
from pointcam.layers.mask_transformer import MaskTransformer
from pointcam.models.pcbot import BasepcBOT
from pointcam.utils.builders import get_optimizer, group_params
from pointcam.utils.crop import PatchedCloud
from pointcam.utils.losses import DiversityLoss, SparsityLoss
from pointcam.utils.visualize import plot_masks_distribution

matplotlib.use("Agg")


class PointCAM(BasepcBOT):
    def __init__(
        self,
        cfg: PointCAMConfig,
    ):
        super().__init__(cfg=cfg)
        self.n_masks = cfg.masknet.n_masks
        self.sparsity_weight = cfg.model.sparsity_weight
        self.diversity_weight = cfg.model.diversity_weight

        self.sparsity_loss = SparsityLoss(self.n_masks, beta=cfg.model.sparsity_beta)
        self.diversity_loss = DiversityLoss()

        self.masknet = MaskTransformer(**cfg.masknet)

        self.automatic_optimization = False

    @classmethod
    def from_exported_checkpoint(
        cls, checkpoint_path: str, drop_path_rate: Optional[float] = None
    ):
        checkpoint = torch.load(checkpoint_path)
        cfg = checkpoint["config"]
        cfg = OmegaConf.merge(OmegaConf.structured(PointCAMConfig), cfg)

        if drop_path_rate is not None:
            cfg.encoder.drop_path_rate = drop_path_rate

        model = cls(cfg=cfg)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        del (
            model.teacher,
            model.ibot_loss,
            model.dino_loss,
            model.sparsity_loss,
            model.diversity_loss,
        )
        return model

    def configure_optimizers(self):
        student_params = group_params(
            self.student, weight_decay=self.cfg.optimizer.weight_decay
        )

        enc_opt, enc_sch = get_optimizer(
            student_params,
            total_steps=self.trainer.estimated_stepping_batches,
            cfg=self.cfg.optimizer,
        )

        masknet_params = group_params(
            self.masknet, weight_decay=self.cfg.optimizer_mask.weight_decay
        )

        mask_opt, mask_sch = get_optimizer(
            masknet_params,
            total_steps=self.trainer.estimated_stepping_batches,
            cfg=self.cfg.optimizer_mask,
        )

        return (
            {
                "optimizer": enc_opt,
                "lr_scheduler": {"scheduler": enc_sch, "name": "lr-encoder"},
            },
            {
                "optimizer": mask_opt,
                "lr_scheduler": {"scheduler": mask_sch, "name": "lr-mask-encoder"},
            },
        )

    def flip_gradient(self, encoder_state: bool):
        for param in self.student.parameters():
            param.requires_grad = encoder_state

        for param in self.masknet.parameters():
            param.requires_grad = not encoder_state

    def forward(self, xyz: PatchedCloud):
        return self.student(xyz, pool=True)

    def mask_forward(self, xyz: PatchedCloud):
        batch_size = xyz.patches.shape[0]

        masks = self.masknet(xyz)
        masks = rearrange(masks, "(b c) p m -> m b c p 1", b=batch_size, m=self.n_masks)

        return masks

    def encoder_forward(self, encoder, xyz: PatchedCloud, mask=None):
        batch_size = xyz.patches.shape[0]

        if mask is not None:
            mask = rearrange(mask, "b c p h -> (b c) p h")

        encoder_cls, encoder_patch = encoder(xyz, mask=mask)

        encoder_cls = rearrange(encoder_cls, "(b c) h-> c b h", b=batch_size)
        encoder_patch = rearrange(encoder_patch, "(b c) p h-> c b p h", b=batch_size)
        return encoder_cls, encoder_patch

    def mpm_step(self, batch, mask_stage: bool = False):
        local_xyz, global_xyz = batch["local_xyz"], batch["global_xyz"]

        global_masks = self.mask_forward(global_xyz)

        # global_teacher & local_student are not influenced by masks
        with torch.no_grad():
            global_teacher_outputs = self.encoder_forward(self.teacher, global_xyz)

        if not mask_stage:
            loc_student_cls, loc_student_patch = self.encoder_forward(
                self.student, local_xyz
            )

        ibot_loss, dino_loss = 0.0, 0.0
        for mask_idx in range(self.n_masks):
            glb_mask = global_masks[mask_idx]
            if not mask_stage:
                glb_mask = glb_mask.detach()

            global_student_outputs = self.encoder_forward(
                self.student,
                global_xyz,
                mask=glb_mask,
            )

            glb_student_cls, glb_student_patch = global_student_outputs
            glb_teacher_cls, glb_teacher_patch = global_teacher_outputs
            if not mask_stage:
                all_student_cls = torch.cat([glb_student_cls, loc_student_cls], dim=0)

            glb_mask = rearrange(glb_mask, "b c p h -> c b p h")

            with torch.autocast(enabled=False, dtype=torch.float32, device_type="cuda"):
                ibot_loss += self.ibot_loss(
                    glb_student_patch, glb_teacher_patch, glb_mask
                )
                if not mask_stage:
                    dino_loss += self.dino_loss(all_student_cls, glb_teacher_cls)
                else:
                    dino_loss += self.dino_loss(glb_student_cls, glb_teacher_cls)

        loss_terms = {
            "ibot": ibot_loss / global_masks.shape[0],
            "dino": dino_loss / global_masks.shape[0],
        }

        if mask_stage:
            global_masks = rearrange(global_masks, "n b c p h -> n (b c) p h")

            loss_terms["sparsity"] = self.sparsity_loss(global_masks)
            loss_terms["diversity"] = self.diversity_loss(global_masks)

        return loss_terms

    def training_step(self, batch, batch_idx):
        # optimizers and lr schedulers
        enc_opt, mask_opt = self.optimizers()
        enc_sch, mask_sch = self.lr_schedulers()

        # inference
        self.flip_gradient(True)
        enc_opt.zero_grad()
        enc_loss_terms = self.mpm_step(batch, mask_stage=False)
        enc_loss = (
            self.ibot_weight * enc_loss_terms["ibot"]
            + self.dino_weight * enc_loss_terms["dino"]
        )

        self.manual_backward(enc_loss)

        if self.cfg.train.clip is not None:
            self.clip_gradients(
                enc_opt,
                gradient_clip_val=self.cfg.train.clip,
                gradient_clip_algorithm="norm",
            )

        enc_opt.step()
        enc_sch.step()

        # masking
        self.flip_gradient(False)
        mask_opt.zero_grad()
        mask_loss_terms = self.mpm_step(batch, mask_stage=True)
        mask_loss = (
            self.ibot_weight * mask_loss_terms["ibot"]
            + self.dino_weight * mask_loss_terms["dino"]
            - self.sparsity_weight * mask_loss_terms["sparsity"]
            - self.diversity_weight * mask_loss_terms["diversity"]
        )

        self.manual_backward(-mask_loss)

        if self.cfg.train.clip is not None:
            self.clip_gradients(
                mask_opt,
                gradient_clip_val=self.cfg.train.clip,
                gradient_clip_algorithm="norm",
            )

        mask_opt.step()
        mask_sch.step()

        self.update_teacher()
        self.teacher_temp_sch.step()

        # logging
        enc_loss_terms["loss"] = enc_loss
        self.log(
            "teacher-temp",
            self.teacher_temp_sch.get(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log_dict(
            {f"train/{key}": val for key, val in enc_loss_terms.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log_dict(
            {
                f"mask/{key}": val
                for key, val in mask_loss_terms.items()
                if key in ["sparsity", "diversity"]
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

    def log_masks(self, xyz: PatchedCloud, masks_probs: torch.Tensor):
        batch_size = xyz.patches.shape[0]
        masks = masks_probs.argmax(dim=0).numpy()

        point_clouds = []
        masks_dists = []
        for idx in range(batch_size):
            points = xyz.patches_decentered[idx].squeeze().detach().cpu()

            split_points = []
            for patch_idx in range(points.shape[0]):
                patch_points = points[patch_idx]

                color = COLORS[masks[idx, patch_idx]]
                color_rgb = torch.Tensor(ImageColor.getrgb(color)).unsqueeze(0)
                color_rgb = color_rgb.repeat(patch_points.shape[0], 1)

                patch_points = torch.cat([patch_points, color_rgb], dim=1)
                split_points.append(patch_points)

            split_points = torch.cat(split_points, dim=0).numpy()

            masks_dist_plot = plot_masks_distribution(
                masks_probs[:, idx].numpy(),
            )

            masks_dists.append(wandb.Image(masks_dist_plot, mode="RGB"))

            point_clouds.append(
                wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": split_points,
                    }
                )
            )

        return point_clouds, masks_dists

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.eval()

        if self.should_log_media(self.current_epoch):
            classes = np.array(self.trainer.datamodule.test_linear_dataset.classes)
            point_clouds = []
            masks_distributions = []
            labels = []

            for batch_idx, batch in enumerate(
                self.trainer.datamodule.test_linear_dataloader()
            ):
                if batch_idx == self.cfg.train.log_masks_n_batches:
                    break

                with torch.no_grad():
                    xyz = batch["xyz"].to(self.device)

                    masks = self.mask_forward(xyz).detach().cpu()  # m b c p 1
                    masks = rearrange(masks, "m b c p 1 -> m (b c) p")

                batch_point_clouds, batch_masks_dists = self.log_masks(xyz, masks)

                point_clouds.extend(batch_point_clouds)
                masks_distributions.extend(batch_masks_dists)
                labels.append(batch["label"])

            labels = classes[torch.cat(labels).numpy()]

            table_cols = ["masked point cloud", "masks distribution", "label"]
            table_data = list(zip(point_clouds, masks_distributions, labels))
            wandb.log({"Masks": wandb.Table(columns=table_cols, data=table_data)})

        self.train()
