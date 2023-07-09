import torch
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

from pointcam.configs import pcBOTConfig
from pointcam.configs.types import FFNLayer
from pointcam.layers import TinyPointNet, Transformer, iBOTHead
from pointcam.models.base import BaseModel, init_weights
from pointcam.utils.builders import get_optimizer, group_params
from pointcam.utils.crop import PatchedCloud, knn_points
from pointcam.utils.losses import DINOLoss, iBOTLoss
from pointcam.utils.schedulers import CosineAnneal, LinearAnneal


def remove_dropout(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            remove_dropout(module)

        if isinstance(module, (nn.Dropout, DropPath)):
            setattr(model, n, nn.Identity())


class Encoder(nn.Module):
    def __init__(
        self,
        n_heads: int = 6,
        depth: int = 4,
        embedding_dim: int = 384,
        projection_dim: int = 512,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 128,
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        shared_head: bool = True,
        ffn_layer: FFNLayer = FFNLayer.SWIGLU,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pointnet = TinyPointNet(embedding_dim)

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embedding_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.transformer = Transformer(
            embedding_dim,
            depth,
            n_heads,
            drop_path_rate=dpr,
            drop_rate=drop_rate,
            ffn_layer=ffn_layer,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.cls_pos_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.norm = nn.LayerNorm(embedding_dim)
        self.projector = iBOTHead(
            embedding_dim,
            projection_dim,
            patch_out_dim=projection_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            shared_head=shared_head,
        )

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos_token, std=0.02)

    @classmethod
    def init_student_teacher(cls, *args, **kwargs):
        student, teacher = cls(*args, **kwargs), cls(*args, **kwargs)
        student.apply(init_weights)
        teacher.load_state_dict(student.state_dict())

        for p in teacher.parameters():
            p.requires_grad = False

        remove_dropout(teacher)
        return student, teacher

    def forward(
        self, xyz: PatchedCloud, mask=None, pool=False, return_hidden_states=False
    ):
        xyz = xyz.flatten()

        embeddings = self.pointnet(xyz)
        pos_embeddings = self.pos_embedding(xyz.centers)

        if mask is not None:
            mask_token = self.mask_token.expand(
                embeddings.size(0), embeddings.size(1), -1
            )
            embeddings = embeddings * (1 - mask) + mask_token * mask

        cls_token = self.cls_token.expand(embeddings.size(0), -1, -1)
        cls_pos_token = self.cls_pos_token.expand(embeddings.size(0), -1, -1)

        embeddings = torch.cat((cls_token, embeddings), dim=1)
        pos_embeddings = torch.cat((cls_pos_token, pos_embeddings), dim=1)

        if return_hidden_states:
            return self.transformer(
                embeddings, pos_embeddings, return_hidden_states=True
            )
        else:
            features = self.transformer(embeddings, pos_embeddings)

        features = self.norm(features)

        if pool:
            cls_features = features[:, 0]
            patch_features = features[:, 1:].max(1).values + features[:, 1:].mean(1)

            return torch.cat([cls_features, patch_features], dim=-1)
        else:
            with torch.autocast(enabled=False, dtype=torch.float32, device_type="cuda"):
                return self.projector(features)


class BasepcBOT(BaseModel):
    def __init__(
        self,
        cfg: pcBOTConfig,
    ):
        super().__init__(cfg=cfg)
        self.ibot_weight = cfg.model.ibot_weight
        self.dino_weight = cfg.model.dino_weight

        self.momentum_sch = CosineAnneal(
            *self.cfg.model.teacher_momentum,
            total_steps=None,
        )

        self.teacher_temp_sch = LinearAnneal(
            *self.cfg.model.teacher_temp, total_steps=None
        )

        self.ibot_loss = iBOTLoss(
            projection_dim=self.cfg.encoder.projection_dim,
            student_temp=self.cfg.model.student_temp,
            teacher_temp_sch=self.teacher_temp_sch,
            center_momentum=self.cfg.model.patch_center_momentum,
        )
        self.dino_loss = DINOLoss(
            projection_dim=self.cfg.encoder.projection_dim,
            student_temp=self.cfg.model.student_temp,
            teacher_temp_sch=self.teacher_temp_sch,
            center_momentum=self.cfg.model.cls_center_momentum,
        )

        self.student, self.teacher = Encoder.init_student_teacher(**cfg.encoder)

    def on_train_start(self):
        super().on_train_start()
        total_steps = self.trainer.estimated_stepping_batches
        step = self.trainer.global_step // 2

        self.momentum_sch.set_total_steps(total_steps)
        self.momentum_sch.set_current_step(step)
        self.teacher_temp_sch.set_total_steps(total_steps)
        self.teacher_temp_sch.set_current_step(step)

    def update_teacher(self):
        m = self.momentum_sch.step().get()
        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.detach().data * (1.0 - m))

        self.log(
            "momentum", m, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )


class pcBOT(BasepcBOT):
    def mask_patches_block(self, xyz):
        xyz = xyz.to("cpu")
        min_ratio, max_ratio = self.cfg.model.masking_ratios
        patches = xyz.patches
        centers = xyz.centers
        n_crops = patches.shape[1]

        patches = rearrange(patches, "b c p k h-> (b c) p k h")
        centers = rearrange(centers, "b c p h-> (b c) p h")
        batch_size, n_patches = patches.shape[0], patches.shape[1]

        centers_idx = torch.multinomial(torch.ones(batch_size, n_patches), 1)
        mask_centers = centers[torch.arange(batch_size).unsqueeze(1), centers_idx]

        mask_length = torch.randint(
            int(min_ratio * n_patches),
            int(max_ratio * n_patches),
            (1,),
        )

        idx = knn_points(
            mask_centers,
            centers,
            K=mask_length.item(),
        ).idx.squeeze(1)

        mask = torch.scatter(torch.zeros(batch_size, n_patches), 1, idx, 1)
        return rearrange(mask, "(b c) p-> b c p 1", c=n_crops).cuda()

    def mask_patches(self, patches):
        min_ratio, max_ratio = self.cfg.model.masking_ratios
        n_crops = patches.shape[1]

        patches = rearrange(patches, "b c p k h-> (b c) p k h")
        batch_size, n_patches = patches.shape[0], patches.shape[1]

        mask = torch.rand(
            batch_size, n_patches, 1, dtype=torch.float32, device=patches.device
        )
        mask = torch.logical_and(mask > min_ratio, mask < max_ratio).float()
        return rearrange(mask, "(b c) p h-> b c p h", c=n_crops)

    def forward(self, xyz: PatchedCloud):
        return self.student(xyz, pool=True)

    def encoder_forward(self, encoder, xyz: PatchedCloud, mask=None):
        batch_size = xyz.patches.shape[0]

        if mask is not None:
            mask = rearrange(mask, "b c p h -> (b c) p h")

        encoder_cls, encoder_patch = encoder(xyz, mask=mask)

        encoder_cls = rearrange(encoder_cls, "(b c) h-> c b h", b=batch_size)
        encoder_patch = rearrange(encoder_patch, "(b c) p h-> c b p h", b=batch_size)
        return encoder_cls, encoder_patch

    def mpm_step(self, batch, batch_idx):
        local_xyz, global_xyz = batch["local_xyz"], batch["global_xyz"]

        # global_mask = self.mask_patches(global_xyz.patches)
        global_mask = self.mask_patches_block(global_xyz)
        global_student_outputs = self.encoder_forward(
            self.student,
            global_xyz,
            mask=global_mask,
        )
        local_student_outputs = self.encoder_forward(self.student, local_xyz)
        with torch.no_grad():
            global_teacher_outputs = self.encoder_forward(self.teacher, global_xyz)

        loc_student_cls, loc_student_patch = local_student_outputs
        glb_student_cls, glb_student_patch = global_student_outputs
        glb_teacher_cls, glb_teacher_patch = global_teacher_outputs
        all_student_cls = torch.cat([glb_student_cls, loc_student_cls], dim=0)

        global_mask = rearrange(global_mask, "b c p h -> c b p h")

        ibot = self.ibot_loss(glb_student_patch, glb_teacher_patch, global_mask)
        dino = self.dino_loss(all_student_cls, glb_teacher_cls)

        loss = self.ibot_weight * ibot + self.dino_weight * dino
        return {
            "loss": loss,
            "ibot": ibot,
            "dino": dino,
        }

    def training_step(self, batch, batch_idx):
        loss_terms = self.mpm_step(batch, batch_idx)

        self.update_teacher()
        self.teacher_temp_sch.step()

        self.log(
            "teacher-temp",
            self.teacher_temp_sch.get(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log_dict(
            {f"train/{key}": val for key, val in loss_terms.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss_terms["loss"]

    def configure_optimizers(self):
        params = group_params(self, weight_decay=self.cfg.optimizer.weight_decay)

        optimizer, scheduler = get_optimizer(
            params,
            total_steps=self.trainer.estimated_stepping_batches,
            cfg=self.cfg.optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
