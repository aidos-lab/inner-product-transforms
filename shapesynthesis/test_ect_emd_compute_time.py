"""
This script evaluates the inference times of the EMD.

"""

from typing import Callable, Tuple, TypeAlias, Any
import torch
import json
from layers.directions import generate_uniform_directions
from metrics.PyTorchEMD import emd

torch.set_float32_matmul_precision("high")

Tensor: TypeAlias = torch.Tensor


# ############################################################################ #
#                                   Constants                                  #
# ############################################################################ #

WARMUP_RUNS = 100
NUM_RERUNS = 1000
NUM_PTS = 2048
RESOLUTION = 64
DTYPE = torch.float32
SCALE = torch.tensor(RESOLUTION // 2).cuda()


# Generate the uniform direction used in all experiments.
v = generate_uniform_directions(
    num_thetas=RESOLUTION,
    d=3,
    seed=2025,
).cuda()


# Set of lin space for experiments.
lin = torch.linspace(start=-2, end=2, steps=RESOLUTION, device="cuda:0").view(-1, 1, 1)


# Arguments for the functions.
ect_kwargs_with_lin = {
    "v": v,
    "radius": 2,
    "resolution": RESOLUTION,
    "scale": SCALE,
}

ect_kwargs_without_lin = {
    "v": v,
    "lin": lin,
    "scale": SCALE,
}

# ############################################################################ #
#                                   Functions                                  #
# ############################################################################ #


def timed(fn: Callable) -> Tuple[Any, float]:
    """Times a function that runs on a single cuda device.

    Parameters
    ----------
    fn : Callable
        Function to be timed.

    Returns
    -------
    Tuple[Any, float]
        Returns a tuple, the first component is the result of the function and
        the second argument is the time it took to run the function.
    """
    start = torch.Event(device="cuda")
    end = torch.Event(device="cuda")
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def compute_ect_point_cloud_with_lin(
    x: Tensor,
    ect_hat: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: Tensor,
) -> Tensor:
    """
    Computes the ECT of a point cloud. The lin object (the discretization of the
    interval) is defined in the function.

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into. (Is irrelevant for
        this experiment.)
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multipicative factor for the sigmoid function.

    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,N,R] where B is the number of
        point clouds (thus ECT's), N is the number of direction and R is the
        resolution.
    """
    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return torch.nn.functional.mse_loss(ect, ect_hat)


def compute_ect_point_cloud_without_lin(
    x: Tensor,
    ect_hat: Tensor,
    v: Tensor,
    lin: Tensor,
    scale: Tensor,
) -> Tensor:
    """
    Computes the ECT of a point cloud. The lin object (the discretization of the
    interval) is defined in the function.

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    lin: Tensor
        The discretization of the unit interval and defines the grid to evaluate
        the ECT on.
    scale : Tensor
        The multipicative factor for the sigmoid function.

    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,N,R] where B is the number of
        point clouds (thus ECT's), N is the number of direction and R is the
        resolution.
    """
    nh = (x @ v).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return torch.nn.functional.mse_loss(ect, ect_hat)


def run_timing(
    ect_fn: Callable, ectkwargs: dict[str, Any], warmup_runs: int, num_reruns: int
) -> float:
    total_time = 0.0
    # Warmup
    for _ in range(warmup_runs):
        x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
        ect_pred = torch.randn(10, RESOLUTION, RESOLUTION, requires_grad=False).cuda()
        out, elapsed = timed(lambda: ect_fn(x_gt, ect_pred, **ectkwargs))

    # Record the times
    for _ in range(num_reruns):
        x_gt = torch.randn(10, NUM_PTS, 3, requires_grad=True).cuda()
        ect_pred = torch.randn(10, RESOLUTION, RESOLUTION, requires_grad=False).cuda()
        out, elapsed = timed(lambda: ect_fn(x_gt, ect_pred, **ectkwargs))
        total_time += elapsed
    return total_time


def run_emd_timing(warmup_runs: int, num_reruns: int) -> float:
    total_time = 0.0
    # Warmup
    for _ in range(warmup_runs):
        x_pred = torch.rand(size=(10, NUM_PTS, 3), requires_grad=True).cuda()
        x_target = torch.rand(size=(10, NUM_PTS, 3), requires_grad=False).cuda()
        timed(lambda: emd.earth_mover_distance(x_pred, x_target, transpose=False))

    # Record the times
    for _ in range(num_reruns):
        x_pred = torch.rand(size=(10, NUM_PTS, 3), requires_grad=True).cuda()
        x_target = torch.rand(size=(10, NUM_PTS, 3), requires_grad=False).cuda()
        _, elapsed = timed(
            lambda: emd.earth_mover_distance(x_pred, x_target, transpose=False)
        )
        total_time += elapsed
    return total_time


# ############################################################################ #
#                            Start of the experiment                           #
# ############################################################################ #

results = {
    "not_compiled_without_lin": 0.0,
    "not_compiled_with_lin": 0.0,
    "compiled_without_lin": 0.0,
    "compiled_with_lin": 0.0,
    "emd": 0.0,
}

# -------------------------------- Without lin ------------------------------- #

total_time = run_timing(
    compute_ect_point_cloud_without_lin,
    ectkwargs=ect_kwargs_without_lin,
    warmup_runs=WARMUP_RUNS,
    num_reruns=NUM_RERUNS,
)
results["not_compiled_without_lin"] += total_time

# -------------------------------- With lin ------------------------------- #

total_time = run_timing(
    compute_ect_point_cloud_with_lin,
    ectkwargs=ect_kwargs_with_lin,
    warmup_runs=WARMUP_RUNS,
    num_reruns=NUM_RERUNS,
)
results["not_compiled_with_lin"] = total_time


# ----------------------------- Compiled without Lin ------------------------- #

# Compile the ect function
compiled_ect_without_lin = torch.compile(compute_ect_point_cloud_without_lin)

total_time = run_timing(
    compiled_ect_without_lin,
    ectkwargs=ect_kwargs_without_lin,
    warmup_runs=WARMUP_RUNS,
    num_reruns=NUM_RERUNS,
)
results["compiled_without_lin"] = total_time


# ----------------------------- Compiled with lin ---------------------------- #

compiled_ect_with_lin = torch.compile(compute_ect_point_cloud_with_lin)

total_time = run_timing(
    compiled_ect_with_lin,
    ectkwargs=ect_kwargs_with_lin,
    warmup_runs=WARMUP_RUNS,
    num_reruns=NUM_RERUNS,
)
results["compiled_with_lin"] = total_time


# ------------------------------------ EMD ----------------------------------- #

total_time = run_emd_timing(
    warmup_runs=WARMUP_RUNS,
    num_reruns=NUM_RERUNS,
)
results["emd"] = total_time


# ############################################################################ #
#                               Print the results                              #
# ############################################################################ #

print(json.dumps(results, indent=4))
