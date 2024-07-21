"""
Helper function to generate a structured set of directions in 2 or 3 dimensions.
"""

import itertools
import torch


def generate_multiview_directions(num_thetas: int, d: int):
    """
    Generates multiple sets of structured directions in n dimensions.

    We generate sets of directions by embedding the 2d unit circle in d
    dimensions and sample this unit circle in a structured fashion. This
    generates d choose 2 structured directions that are organized in channels,
    compatible with the ECT calculations.

    After computing the ECT, we obtain an d choose 2 channel image where each
    channel consists of a structured ect along a hyperplane. For the 3-d case we
    would obtain a 3 channel ect with direction sampled along the xy, xz and yz
    planes in three dimensions.

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    w = torch.vstack(
        [
            torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas)),
            torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas)),
        ]
    )

    # We obtain n choose 2 channels.
    idx_pairs = list(itertools.combinations(range(d), r=2))

    v = torch.zeros(size=(len(idx_pairs), d, num_thetas))

    for idx, idx_pair in enumerate(idx_pairs):
        v[idx, idx_pair[0], :] = w[0]
        v[idx, idx_pair[1], :] = w[1]

    return v


def generate_directions(num_thetas: int = 64, d: int = 3, device: str = "cpu"):
    """
    Generates a structured seet of directions in either 2 or 3 dimensions.
    If the dimension is two, we evenly space the directions along the unit
    circle.
    In the 3d case we sample along three unit circles, the xy, xz and yz planes.
    Along these unit circles we sample evenly. A check is performed to ensure
    the number of directions provided is divisible by three.
    """

    if d == 3:
        if num_thetas % 3 != 0:
            raise ValueError(
                f"Number of theta's should be divisible by three, you provided {num_thetas}"
            )
        w1 = torch.vstack(
            [
                torch.sin(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.cos(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.zeros_like(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
            ]
        )

        w2 = torch.vstack(
            [
                torch.sin(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.zeros_like(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.cos(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
            ]
        )

        w3 = torch.vstack(
            [
                torch.zeros_like(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.sin(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
                torch.cos(
                    torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
                ),
            ]
        )
        v = torch.hstack([w1, w2, w3])
    elif d == 2:
        v = torch.vstack(
            [
                torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas, device=device)),
                torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas, device=device)),
            ]
        )
    else:
        raise ValueError("d should be either 2 or three")
    return v
