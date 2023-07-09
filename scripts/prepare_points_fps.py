import argparse
import logging
from multiprocessing import Pool

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from pointcam.configs.constants import DATA_PATH
from pointcam.utils.crop import fps

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="ModelNet40")
parser.add_argument("--out-dir", type=str, default="ModelNet40FPS")
parser.add_argument("--ext", type=str, default="off")
parser.add_argument("--n-points", type=int, default=1024)
parser.add_argument("--force", action="store_true", default=False)
parser.add_argument("--n-process", type=int, default=16)
args = parser.parse_args()

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

root_path = DATA_PATH / args.dir
output_path = DATA_PATH / args.out_dir
meshes_paths = list(root_path.glob(f"**/*.{args.ext}"))


def sample_point_cloud(mesh_path):
    new_model_path = mesh_path.parent / (mesh_path.stem + ".npy")
    new_model_path = output_path / new_model_path.relative_to(root_path)
    new_model_path.parent.mkdir(exist_ok=True, parents=True)

    if not new_model_path.exists() or args.force:
        mesh = trimesh.load(mesh_path, force="mesh")
        point_cloud = mesh.sample(min(args.n_points * 8, 16384))
        point_cloud = torch.from_numpy(point_cloud).float().unsqueeze(0)
        point_cloud = fps(point_cloud, args.n_points)
        point_cloud = point_cloud.squeeze(0).numpy()

        np.save(new_model_path, point_cloud)

        return True
    return False


if args.n_process > 0:
    p = Pool(args.n_process)
    results = []
    for result in tqdm(
        p.imap_unordered(sample_point_cloud, meshes_paths), total=len(meshes_paths)
    ):
        results.append(result)
else:
    results = [sample_point_cloud(mesh_path) for mesh_path in tqdm(meshes_paths)]

print(f"Done. {sum(results)} files were created.")
