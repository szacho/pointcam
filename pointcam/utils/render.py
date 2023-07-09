import json
import typing as t
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import ImageColor

from pointcam.configs.constants import COLORS


def decode_image(byte_data: t.List[float]) -> np.ndarray:
    byte_data = np.asarray(byte_data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
    return img


def standardize_bbox(pcl: np.ndarray, points_per_object: int) -> np.ndarray:
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[:points_per_object]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.amax(maxs - mins)
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="1000"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="640"/>
                <integer name="height" value="480"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <float name="alpha" value="0.1"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="0.63,0.61,0.58"/> <!-- default 0.5 -->
        </bsdf>

        <shape type="rectangle">
            <transform name="toWorld">
                <lookat origin="0,-11,5" target="0,0,0" />
                <scale x="0.5" y="0.5" z="0.5" />
            </transform>
            <emitter type="area">
                <spectrum name="radiance" value="40"/>
            </emitter>
        </shape>

    """

xml_ball_segment = """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="roughplastic">
                <string name="distribution" value="ggx"/>
                <float name="intIOR" value="1.61"/>
                <float name="alpha" value="0.9"/>
                <rgb name="diffuseReflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="20" y="20" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="20" y="20" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def process_single(points: np.ndarray, port: int, is_rotated: bool) -> np.ndarray:
    if points.shape[-1] == 4:
        points, labels = points[:, :3], points[:, 3].astype(np.uint8)

    points = standardize_bbox(points, points.shape[0])

    if not is_rotated:
        points = points[:, [2, 0, 1]]
        points[:, 0] *= -1
    else:
        points[:, 1] *= -1
        points = points[:, [1, 0, 2]]

    points[:, 2] += 0.0125
    xml_segments = [xml_head]

    for i in range(points.shape[0]):
        color = ImageColor.getcolor(COLORS[labels[i]], "RGB")
        color = colormap(color[0] / 255, color[1] / 255, color[2] / 255)

        xml_segments.append(
            xml_ball_segment.format(points[i, 0], points[i, 1], points[i, 2], *color)
        )

    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)
    result = requests.post(f"http://localhost:{port}/render", data=xml_content)
    data = json.loads(result.content)
    an_img = decode_image(data)
    return an_img


def visualize(
    path: str,
    is_torch: bool,
    port: int,
    is_rotated: bool,
):
    path = Path(path)
    if is_torch:
        points = torch.load(path).detach().cpu().numpy()
    else:
        points = np.load(path)

    img = process_single(points, port, is_rotated)
    out_image_path = path.parent / (path.stem + ".png")
    cv2.imwrite(str(out_image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
