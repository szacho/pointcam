import torch
from torch import nn

from pointcam.configs.types import FFNLayer
from pointcam.layers.pointnet import TinyPointNet
from pointcam.layers.transformer import Transformer
from pointcam.models.base import init_weights
from pointcam.utils.crop import PatchedCloud


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_heads: int = 3,
        depth: int = 2,
        embedding_dim: int = 384,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.1,
        n_patches: int = 64,
        n_masks: int = 3,
        masks_temperature: float = 0.02,
        ffn_layer: FFNLayer = FFNLayer.SWIGLU,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.n_masks = n_masks
        self.temp = masks_temperature

        self.pointnet = TinyPointNet(embedding_dim)

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embedding_dim)
        )

        self.transformer = Transformer(
            embedding_dim,
            depth,
            n_heads,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            ffn_layer=ffn_layer,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.masknet = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_masks, bias=False),
        )
        self.activation = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self, xyz: PatchedCloud):
        xyz = xyz.flatten()

        embeddings = self.pointnet(xyz)
        pos_embeddings = self.pos_embedding(xyz.centers)

        features = self.transformer(embeddings, pos_embeddings)
        features = self.norm(features)

        with torch.autocast(enabled=False, dtype=torch.float32, device_type="cuda"):
            logits = self.masknet(features) / self.temp
            probs = self.activation(logits)

        return probs
