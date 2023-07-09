# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: opus
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import ipywidgets as widgets
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from pointcam.configs import types as t
from pointcam.configs.base import DataConfig
from pointcam.datasets import ModelNetDataModule
from pointcam.models import PointCAM
from pointcam.utils.render import visualize
from pointcam.utils.visualize import COLORS

pl.seed_everything(42)
plt.ion()

# %%
CHECKPOINT = "checkpoints/pointcam.pt"


# %% [markdown]
# ## load model and data

# %%
model = PointCAM.from_exported_checkpoint(CHECKPOINT)
cfg = DataConfig(
    dataset=t.Dataset.MODELNET40FPS,
    batch_size=32,
    n_patches=64,
    points_per_patch=32,
    n_points=1024,
    n_global_crops=0,
    n_local_crops=0,
    uniform=False,
)

datamodule = ModelNetDataModule(cfg)
datamodule.setup()
CLASSES = datamodule.test_dataset.classes


# %% [markdown]
# ## generate masks

# %%
model = model.eval().cuda()

test_masks, test_xyz, test_labels = [], [], []

for batch in tqdm(datamodule.test_dataloader()):
    xyz, labels = batch["xyz"], batch["label"]
    xyz = xyz.to("cuda")

    with torch.no_grad():
        masks = model.mask_forward(xyz).squeeze(2)

    test_masks.append(masks)
    test_labels.append(labels)
    xyz_patched = xyz.patches_decentered
    test_xyz.append(xyz_patched)

test_masks = (
    torch.cat(test_masks, dim=1).permute(1, 0, 2, 3).squeeze(-1).cpu()
)  # first dim is n_masks, second is batch_size
test_xyz = (
    torch.cat(test_xyz, dim=0).squeeze(1).cpu()
)  # first dim is batch_size, second is crop_dim
test_labels = torch.cat(test_labels, dim=0)


# %% [markdown]
# ## visualize masks

# %%
# %matplotlib widget


class MasksPlot(widgets.Box):
    def __init__(self, xyz, masks, labels):
        super().__init__()
        output = widgets.Output()
        self.output_path = Path("images")
        self.output_path.mkdir(exist_ok=True)

        indices = torch.argsort(labels)

        self.xyz = xyz[indices]
        self.masks = masks[indices]
        self.labels = labels[indices]

        self.n_masks = masks.shape[1]

        with output:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(8, 8), subplot_kw={"projection": "3d"}
            )

        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False

        self.int_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.xyz) - 1,
            step=1,
            description="Index:",
        )
        self.int_slider.style.handle_color = "lightblue"
        self.export_button = widgets.Button(
            description="RENDER", button_style="warning"
        )

        self.plot_base(self.int_slider.value)

        controls = widgets.HBox([self.int_slider, self.export_button])
        self.int_slider.observe(self.update_plot, names="value")
        self.export_button.on_click(self.export)

        self.children = [controls, output]

    def export(self, change):
        index = self.int_slider.value

        xyz = self.xyz[index]
        label = self.labels[index]
        masks = self.masks[index].argmax(dim=0).unsqueeze(1)

        torch.unique(xyz, dim=0)

        masks = masks.expand(-1, xyz.shape[1]).unsqueeze(-1).float()

        labeled_xyz = torch.cat([xyz, masks], dim=-1).flatten(0, 1)
        unique_scores = [0]
        unique_labeled_points = []

        labeled_xyz = torch.flip(labeled_xyz, dims=[0])

        # remove duplicated points to fix color mixing
        for point in labeled_xyz:
            score = point[:3].sum() + point[:3].prod()  # should be MSE
            score_diffs = [abs(score - s) for s in unique_scores]
            if min(score_diffs) > 1e-8:
                unique_labeled_points.append(point)
                unique_scores.append(score)

        labeled_xyz = torch.stack(unique_labeled_points)

        points_path = self.output_path / f"{index}_{CLASSES[label]}.pt"
        torch.save(labeled_xyz, points_path)

        try:
            visualize(points_path, is_torch=True, port=8000, is_rotated=True)
        except ConnectionError as e:
            print("Mitsuba is not running!")
            raise e
        finally:
            os.remove(points_path)

    def update_plot(self, change):
        self.ax.clear()
        self.plot_base(change.new)

    def plot_base(self, index):
        xyz = self.xyz[index]
        masks = self.masks[index]
        label = self.labels[index]

        masks = masks.argmax(dim=0)

        for patch_idx, mask_idx in enumerate(masks):
            self.ax.scatter(
                xyz[patch_idx, :, 0],
                xyz[patch_idx, :, 1],
                xyz[patch_idx, :, 2],
                c=COLORS[mask_idx],
                s=5,
            )

        self.ax.set_aspect("equal")
        self.ax.set_title(f"{CLASSES[label]}")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_axis_off()


MasksPlot(test_xyz, test_masks, test_labels)


# %%
