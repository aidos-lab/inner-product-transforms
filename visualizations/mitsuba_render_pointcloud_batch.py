import sys

from yaml import emit
from mitsuba import Thread, Bitmap, Struct

# patch work.
# sys.path.insert(0, r"C:\Users\ernst\mitsuba3\build\Release\python")

import torch

import mitsuba as mi
from matplotlib import pyplot as plt
import numpy as np

mi.set_variant("cuda_rgb")

COLOR = [0.1, 0.27, 0.86]

STATE_DICT = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": -1, "hide_emitters": True},
    "sensor": {
        "type": "perspective",
        "far_clip": 100,
        "near_clip": 0.1,
        "to_world": mi.ScalarTransform4f().look_at(
            origin=[-3, 3, 3], target=[0, 0, -0.1], up=[0, 0, 1]
        ),
        "fov": 12,
        "film": {
            "type": "hdrfilm",
            "pixel_format": "rgba",
            "component_format": "float32",
            "width": 512,
            "height": 512,
            "rfilter": {
                "type": "gaussian",
            },
        },
    },
    "emitter_plane": {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f()
        .scale([10, 10, 1])
        .look_at(origin=[0.5, 0.5, 20], target=[0, 0, 0.0], up=[0, 0, 1]),
        "emitter": {
            "type": "area",
            "radiance": {
                "type": "rgb",
                "value": 6,
            },
        },
    },
}

ground_plane = {
    "type": "rectangle",
    "bsdf": {
        "type": "roughplastic",
        "distribution": "ggx",
        # "int_ior": 1.46,
        "diffuse_reflectance": {"type": "rgb", "value": [1, 1, 1]},
    },
}

sphere_template = {
    "type": "sphere",
    "radius": 0.015,
    "bsdf": {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": COLOR},
    },
}

emitter = {
    "type": "constant",
    "radiance": {
        "type": "rgb",
        "value": 0.1,
    },
}


def standardize_pc(point_clouds, scale=1.0):
    """"""
    center = np.mean(point_clouds, axis=1, keepdims=True)
    scale = np.amax(point_clouds - np.amin(point_clouds, axis=1, keepdims=True))
    scaled = ((point_clouds - center) / scale).astype(np.float32)
    # Axis reshuffle happens after normalize, "z-axis" is the second variable.
    scaled = scaled * scale
    # # Put lowest point at 0
    # z_min = np.amin(scaled, axis=1, keepdims=True)
    # z_min[:, :, 0] = 0
    # z_min[:, :, 2] = 0
    # scaled = scaled - z_min

    # z_max = np.amax(scaled, axis=1, keepdims=True)
    # z_max[z_max < 0.4] = 1.0  # No scaling
    # scaled = scaled / (z_max * 2.5)
    return scaled


def create_state_dict(point_cloud, show_background=True):

    z_min = np.amin(point_cloud, axis=0)[-1]

    if show_background:
        ground_plane["to_world"] = (
            mi.ScalarTransform4f().translate([0, 0, z_min - 0.1]).scale([10, 10, 1])
        )
        STATE_DICT["ground_plane"] = ground_plane.copy()
        STATE_DICT["emitter"] = emitter.copy()
    else:
        emitter_extra = {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": 0.4,
            },
        }
        STATE_DICT["emitter"] = emitter_extra.copy()

    spheres = {}

    for i, pt in enumerate(point_cloud):
        x, y, z = pt
        sphere_template["center"] = [x, y, z]
        spheres[f"sphere_{i}"] = sphere_template.copy()

    state_dict = STATE_DICT | spheres
    return state_dict


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("INPUT", type=str, help="Input pointclouds")
parser.add_argument("--idx", nargs="+", default=[0], type=int)
parser.add_argument("--hide-bg", action="store_true", default=False)
args = parser.parse_args()

output_path = args.INPUT.split("\\")[2]
output_name = args.INPUT.split("\\")[-1].split(".")[0]
cate = []
for x in args.INPUT.split("\\"):
    for y in x.split("_"):
        cate.append(y)

point_clouds = torch.load(args.INPUT).cpu().numpy()

print(cate)
# Rescale only chairs.
if "chair" in cate:
    scale = 0.8
    sphere_template["radius"] *= 0.8
else:
    scale = 1.0

print("SCALE", scale)

# # Assume of shape [-1,2045,3]
pts_normalized = standardize_pc(point_clouds, scale=scale)
pts_normalized *= scale
point_cloud = pts_normalized[:, :, [0, 2, 1]]


# Make the object face us. That is swap the x-axis.
point_cloud[:, :, 1] *= -1

print(point_cloud.shape)

for i in args.idx:
    show_bg = not args.hide_bg
    state_dict = create_state_dict(point_cloud[i], show_background=show_bg)

    suffix = ""
    if args.hide_bg:
        suffix = "_nobg"
    if len(args.idx) == 1:
        filename = "./figures/" + output_path + f"/render{suffix}.png"
    else:
        filename = f"./figures/{output_path}/{output_name}_render_{i}{suffix}.png"

    scene = mi.load_dict(state_dict)
    img = mi.render(scene, spp=512)
    bmp = mi.Bitmap(img)
    bmp_small = bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, True)
    print("Saving to", filename)
    bmp_small.write(
        filename,
    )
