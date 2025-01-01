"""
Script to compute the average time it takes to render a point cloud using our
rendering method. The configurations are equal to the ones used in the
experiments.
"""

from typing import Any, Callable, Tuple
import torch
from layers.directions import generate_uniform_directions
from layers.ect import compute_ect_point_cloud
from renderers.pointcloud import render_point_cloud

torch.set_float32_matmul_precision("medium")

# ############################################################################ #
#                                   Constants                                  #
# ############################################################################ #

NUM_EPOCHS = 2000
RESOLUTION = 128
SCALE = int(RESOLUTION * 0.25)
IDX = 8
SEED = 2013
RADIUS = torch.tensor(7)
CATE = "car"
RADIUS = 7
BATCH_SIZE = 32


# Load the ground truth point clouds.
# We only load 10 point clouds.
x_gt = torch.load("./results/encoder_ect_airplane/references.pt", weights_only=True)[
    :10
]


# Initialize the random noise vector.
x_init = RADIUS / 2 * (torch.rand_like(x_gt) - 0.5)


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


# ############################################################################ #
#                                Run Experiment                                #
# ############################################################################ #

for resolution in [32, 64, 128]:

    render_point_cloud_compiled = torch.compile(render_point_cloud)

    # Generate the set of directions.
    v = generate_uniform_directions(num_thetas=resolution, d=3, seed=SEED).cuda()

    # Compute the ground truth ect.
    ect_gt = compute_ect_point_cloud(
        x_gt, v, radius=RADIUS, resolution=resolution, scale=SCALE
    )

    # Warmup
    for _ in range(10):
        x_rendered, elapsed = timed(
            lambda: render_point_cloud_compiled(
                x_init,
                ect_gt,
                v,
                NUM_EPOCHS,
                scale=SCALE,
                radius=RADIUS,
                resolution=resolution,
            )
        )

    # Run the experiment.
    times = 0.0
    for _ in range(10):
        x_rendered, elapsed = timed(
            lambda: render_point_cloud_compiled(
                x_init,
                ect_gt,
                v,
                NUM_EPOCHS,
                scale=SCALE,
                radius=RADIUS,
                resolution=resolution,
            )
        )
        times += elapsed

    print(f"Average time over 10 runs with resolution {resolution} :", times / 10)
