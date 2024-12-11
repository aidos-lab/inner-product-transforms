import numpy as np
from nicegui import ui


def generate_data(frequency: float = 1.0):
    x, y = np.meshgrid(np.linspace(-3, 3), np.linspace(-3, 3))
    z = np.sin(x * frequency) * np.cos(y * frequency) + 1
    points = np.dstack([x, y, z]).reshape(-1, 3)
    colors = points / [6, 6, 2] + [0.5, 0.5, 0]
    return points, colors


with ui.scene().classes("w-full h-64") as scene:
    points, colors = generate_data()
    point_cloud = scene.point_cloud(points, colors, point_size=0.05)

ui.slider(min=0.1, max=3, step=0.1, value=1).on_value_change(
    lambda e: point_cloud.set_points(*generate_data(e.value))
)

ui.run()
