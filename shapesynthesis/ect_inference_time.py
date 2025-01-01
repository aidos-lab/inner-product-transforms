from typing import TypeAlias

import torch
from layers.directions import generate_uniform_directions

torch.set_float32_matmul_precision("medium")

Tensor: TypeAlias = torch.Tensor


def compute_ect_point_cloud_with_lin(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: Tensor,
) -> Tensor:
    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return ect


def compute_ect_point_cloud(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    lin: Tensor,
    scale: Tensor,
) -> Tensor:
    nh = (x @ v).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return ect


NUM_RERUNS = 10
NUM_PTS = 1024
RESOLUTION = 128
DTYPE = torch.float16
SCALE = torch.tensor(RESOLUTION // 2).cuda()
# SCALE = 20
v = generate_uniform_directions(
    num_thetas=RESOLUTION,
    d=3,
    seed=2025,
).cuda()

# Load the ECT's
x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()

lin = torch.linspace(start=-2, end=2, steps=2, device="cuda:0").view(-1, 1, 1)


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


for _ in range(NUM_RERUNS):
    x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
    out, elapsed = timed(
        lambda: compute_ect_point_cloud(
            x_gt, v, radius=2, resolution=RESOLUTION, lin=lin, scale=SCALE
        )
    )


print("NOT COMPILED WITHOUT LIN")

for _ in range(NUM_RERUNS):
    x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
    out, elapsed = timed(
        lambda: compute_ect_point_cloud(
            x_gt, v, radius=2, resolution=RESOLUTION, lin=lin, scale=SCALE
        )
    )

    print(elapsed)


for _ in range(NUM_RERUNS):
    x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
    out, elapsed = timed(
        lambda: compute_ect_point_cloud_with_lin(
            x_gt, v, radius=2, resolution=RESOLUTION, scale=SCALE
        )
    )


print("COMPILED WITHOUT LIN")
compiled_ect = torch.compile(compute_ect_point_cloud)

for _ in range(NUM_RERUNS):
    x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
    out, elapsed = timed(
        lambda: compiled_ect(
            x_gt, v, radius=2, resolution=RESOLUTION, lin=lin, scale=SCALE
        )
    )

    print(elapsed)


print("NOT COMPILED WITH LIN")


for _ in range(NUM_RERUNS):
    x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
    out, elapsed = timed(
        lambda: compute_ect_point_cloud_with_lin(
            x_gt, v, radius=2, resolution=RESOLUTION, scale=SCALE
        )
    )

    print(elapsed)


print("COMPILED WITH LIN")

compiled_ect_with_lin = torch.compile(compute_ect_point_cloud_with_lin)

for _ in range(NUM_RERUNS):
    x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
    out, elapsed = timed(
        lambda: compiled_ect_with_lin(
            x_gt, v, radius=2, resolution=RESOLUTION, scale=SCALE
        )
    )

    print(elapsed)
