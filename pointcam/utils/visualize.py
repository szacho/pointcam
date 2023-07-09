import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from plotly import graph_objects as go

from pointcam.configs.constants import COLORS


def plot_embeddings(*embeddings, labels: list[str]):
    fig = go.Figure()
    for i, embedding in enumerate(embeddings):
        color_idx = i % len(COLORS)
        size = 3
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(color=COLORS[color_idx], size=size),
                name=labels[i],
            )
        )

    fig.update_layout(
        height=600,
        width=1000,
        legend=dict(
            itemsizing="constant",
        ),
    )

    return fig


def plot_masks_distribution(masks: torch.Tensor):
    ind = np.arange(masks.shape[1])
    _ = plt.subplots(figsize=(8, 3))

    bottom = np.zeros(masks.shape[1])
    for idx, mask_probs in enumerate(masks):
        plt.bar(ind, mask_probs, bottom=bottom, color=COLORS[idx])
        bottom += mask_probs

    plt.xlim(-1, masks.shape[1])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.xticks(np.arange(0, 65, step=8), color="gray", fontsize=7)
    plt.yticks(color="gray", fontsize=7)
    plt.gca().spines[["top", "right", "bottom", "left"]].set_visible(False)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight", transparent=True)
    plt.close()

    return Image.open(img_buf)
