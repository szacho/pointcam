import math

import einops
import torch
from torch import linalg as LA
from torch import nn
from torch.nn import functional as F

from pointcam.configs.constants import EPSILON_FP16
from pointcam.utils.schedulers import LinearAnneal


def entropy(teacher_out, student_out, mask=None):
    entropy = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

    if mask is not None:
        entropy = torch.sum(entropy * mask, dim=-1) / torch.sum(mask, dim=-1).clamp(
            min=1.0
        )

    return entropy.mean()


class iBOTLoss(nn.Module):
    def __init__(
        self,
        teacher_temp_sch: LinearAnneal,
        student_temp: float = 0.1,
        projection_dim: int = 512,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp_sch = teacher_temp_sch
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, 1, projection_dim))

    def forward(self, student_patches, teacher_patches, masks=None):
        """
        Args:
            student_patches: (n_crops, batch_size, n_patches, projection_dim)
            teacher_patches: (n_crops, batch_size, n_patches, projection_dim)
            masks: (n_crops, batch_size, n_patches, 1)
        """
        student_patches = student_patches.flatten(0, 1)
        teacher_patches = teacher_patches.flatten(0, 1)
        masks = masks.flatten(0, 1).squeeze(-1)
        teacher_temp = self.teacher_temp_sch.get()

        self.center = self.center.to(student_patches.device)

        student_patches = student_patches / self.student_temp
        teacher_patches = F.softmax(
            (teacher_patches - self.center) / teacher_temp, dim=-1
        ).detach()

        ibot_loss = entropy(teacher_patches, student_patches, masks)
        self.update_center(teacher_patches)
        return ibot_loss

    @torch.no_grad()
    def update_center(self, teacher_patches):
        patches_center = torch.sum(teacher_patches.mean(1), dim=0, keepdim=True)
        patches_center = patches_center / len(teacher_patches)

        self.center = self.center * self.center_momentum + patches_center * (
            1 - self.center_momentum
        )


class DINOLoss(nn.Module):
    def __init__(
        self,
        teacher_temp_sch: LinearAnneal,
        student_temp: float = 0.1,
        projection_dim: int = 512,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp_sch = teacher_temp_sch
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, projection_dim))

    def forward(self, student_cls_tokens, teacher_cls_tokens):
        """
        Args:
            student_cls_tokens: (n_crops, batch_size, projection_dim)
            teacher_cls_tokens: (m_crops, batch_size, projection_dim)
        """
        n_student_crops = len(student_cls_tokens)
        n_teacher_crops = len(teacher_cls_tokens)
        teacher_temp = self.teacher_temp_sch.get()

        self.center = self.center.to(student_cls_tokens.device)

        dino_loss = 0
        for teacher_idx in range(n_teacher_crops):
            for student_idx in range(n_student_crops):
                if teacher_idx == student_idx:
                    continue

                student_cls = student_cls_tokens[student_idx]
                student_cls = student_cls / self.student_temp

                teacher_cls = teacher_cls_tokens[teacher_idx]
                teacher_cls = F.softmax(
                    (teacher_cls - self.center) / teacher_temp, dim=-1
                ).detach()

                dino_loss += entropy(teacher_cls, student_cls)

        n_loss_terms = n_student_crops * n_teacher_crops - min(
            n_student_crops, n_teacher_crops
        )
        dino_loss /= n_loss_terms

        self.update_center(teacher_cls_tokens)
        return dino_loss

    @torch.no_grad()
    def update_center(self, teacher_cls):
        teacher_cls = teacher_cls.flatten(0, 1)

        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        cls_center = cls_center / len(teacher_cls)

        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )


class SparsityLoss(nn.Module):
    """
    Computes the sparsity loss of a masks.
    """

    def __init__(self, n_masks: int, beta: float = 1.5):
        super().__init__()
        self.n_masks = n_masks
        self.beta = beta

    def forward(self, masks: torch.Tensor):
        """
        Args:
            masks: masks of shape (n_masks, batch_size, n_patches, 1)
        """
        masks = einops.rearrange(masks, "n b p 1 -> b n p")
        masks_probs = masks.sum(2) / masks.shape[2]

        loss = 1 / (torch.sin(masks_probs * math.pi) + EPSILON_FP16) - 1
        loss *= math.sin(math.pi * self.n_masks**-self.beta)

        return loss.sum(dim=1).mean()


class DiversityLoss(nn.Module):
    """
    Computes the diversity loss of masks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, masks: torch.Tensor):
        """
        Args:
            masks: masks of shape (n_masks, batch_size, n_patches, 1)
        """
        masks = einops.rearrange(masks, "n b p 1 -> b n p")

        mask_norms = LA.norm(masks, dim=2)
        identity = torch.eye(masks.shape[1], device=masks.device)

        tops = einops.einsum(masks, masks, "b n p, b m p-> b n m")
        bottoms = (
            einops.einsum(mask_norms, mask_norms, "b n, b m-> b n m") + EPSILON_FP16
        )

        loss = ((identity - tops / bottoms) ** 2).mean(dim=(1, 2))
        return loss.mean()
