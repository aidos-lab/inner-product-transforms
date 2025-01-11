from __future__ import annotations
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyvista as pv


# Data to display
rec_pc = torch.load("./results/interpolation/linear_airplane.pt").cpu().numpy()
t = list(range(len(rec_pc)))
# h = np.sin(t)
# v = np.cos(t)

# # Define a Matplotlib figure.
# # Use a tight layout to keep axis labels visible on smaller figures.
# f, ax = plt.subplots(tight_layout=True)
# h_line = ax.plot(t[:1], h[:1])[0]
# ax.set_ylim([-1, 1])
# ax.set_xlim([0, 5])
# ax.set_xlabel("Time (s)")
# _ = ax.set_ylabel("Height (m)")


# Define plotter, add the created matplotlib figure as the first (left) chart
# to the scene, and define a second (right) chart.
p = pv.Plotter()

# # Add first chart
# h_chart = pv.ChartMPL(f, size=(0.46, 0.25), loc=(0.02, 0.06))
# h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
# p.add_chart(h_chart)

# # Second chart
# v_chart = pv.Chart2D(
#     size=(0.46, 0.25),
#     loc=(0.52, 0.06),
#     x_label="Time (s)",
#     y_label="Velocity (m/s)",
# )
# v_line = v_chart.line(t[:1], v[:1])
# v_chart.y_range = (-1, 1)
# v_chart.background_color = (1.0, 1.0, 1.0, 0.4)
# p.add_chart(v_chart)

# Mesh
# p.add_mesh(pv.Sphere(1), name="sphere", render=False)

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
    # k = np.count_nonzero(t < time)
    # h_line.set_xdata(t[: k + 1])
    # h_line.set_ydata(h[: k + 1])
    # v_line.update(t[: k + 1], v[: k + 1])

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
