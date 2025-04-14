"""
Script to plot the full trajectory of a render of an ECT. 
"""

from __future__ import annotations
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyvista as pv


# Data to display
rec_pc = torch.load("./results/rendered_ect/airplane/full_orbit.pt").cpu().numpy()
# rec_pc = torch.load("./render.pt").cpu().numpy()
t = list(range(len(rec_pc)))

p = pv.Plotter()


p.add_points(
    points=rec_pc[0].reshape(-1, 3),
    style="points",
    name="point_cloud",
    color="lightgray",
    render_points_as_spheres=True,
)

# Update.
p.show(auto_close=False, interactive=True, interactive_update=True)


# Method and slider to update all visuals based on the time selection
def update_time(idx_float):
    idx = int(idx_float)

    p.add_points(
        points=rec_pc[idx].reshape(-1, 3),
        style="points",
        name="point_cloud",
        color="lightgray",
        render=False,
        render_points_as_spheres=True,
    )
    p.update()


time_slider = p.add_slider_widget(
    update_time,
    [0, t[-1]],
    0,
    "Time",
    (0.25, 0.9),
    (0.75, 0.9),
    interaction_event="always",
)

# Start incrementing time automatically
for i in t:
    # ax.set_xlim([0, t[i]])
    time_slider.GetSliderRepresentation().SetValue(i)
    update_time(i)
    time.sleep(0.1)

p.show()  # Keep plotter open to let user play with time slider
