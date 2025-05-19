"""
Helper function to generate a structured set of directions in 2 or 3 dimensions.
"""

import torch
import numpy as np


def generate_uniform_directions(num_thetas: int = 64, d: int = 3, seed=None):
    """
    Generate randomly sampled directions from a sphere in d dimensions.

    First a standard gaussian centered at 0 with standard deviation 1 is sampled
    and then projected onto the unit sphere. This yields a uniformly sampled set
    of points on the unit spere. Please note that the generated shapes with have
    shape [d, num_thetas].

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    if not seed:
        raise ValueError("Seed can not be None.")

    rng = np.random.RandomState(seed=seed)
    v = torch.tensor(rng.normal(size=(d, num_thetas)), dtype=torch.float)
    v /= v.pow(2).sum(axis=0).sqrt().unsqueeze(1).T
    return v
