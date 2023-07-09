from timm.models.layers import DropPath
from torch import nn
from xformers.components.feedforward import MLP as xMLP
from xformers.ops import memory_efficient_attention, unbind

from pointcam.configs.types import FFNLayer
from pointcam.layers.swiglu import SwiGLUFFNFused


class Mlp(xMLP):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__(
            dim_model=in_features,
            hidden_layer_multiplier=4,
            dropout=drop,
            activation="gelu",
        )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        attn = memory_efficient_attention(*unbind(qkv, 2), p=self.attn_drop)
        attn = attn.reshape([B, N, C])

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        ffn_layer: FFNLayer = FFNLayer.MLP,
    ):
        super().__init__()

        match ffn_layer:
            case FFNLayer.MLP:
                ffn = Mlp
            case FFNLayer.SWIGLU:
                ffn = SwiGLUFFNFused
            case _:
                raise ValueError(f"Unknown FFN layer: {ffn_layer}")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                    ffn_layer=ffn,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos, return_hidden_states=False):
        if return_hidden_states:
            hidden_states = []

        for _, block in enumerate(self.blocks):
            x = block(x + pos)
            if return_hidden_states:
                hidden_states.append(x)

        if return_hidden_states:
            return hidden_states

        return x
