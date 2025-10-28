import pyvista as pv
import torch

from metrics.evaluation import compute_all_metrics
from plotting import plot_grid, plot_recon_3d

pcs = torch.load("results/unet/pc.pt").cpu().view(-1, 2048, 3)
refs = torch.load("results/encoder_airplane/references.pt").cpu().view(-1, 2048, 3)

means = torch.load("results/encoder_airplane/means.pt").cpu()
stdev = torch.load("results/encoder_airplane/stdevs.pt").cpu()

print(pcs.shape)
print(refs.shape)

print(means.shape)
print(stdev.shape)

# pcs = pcs * stdev[:32] + means[:32]

pl = pv.Plotter(
    shape=(4, 8),
    window_size=[200 * 8, 200 * 4],
    border=False,
    polygon_smoothing=True,
    off_screen=True,
)

for col in range(8):
    for row in range(4):
        # First plat
        pl.subplot(row, col)
        pl.add_points(
            pcs[col + 8 * row].reshape(-1, 3).cpu().numpy(),
            style="points",
            emissive=False,
            show_scalar_bar=False,
            render_points_as_spheres=True,
            color="lightgray",
            point_size=3,
            ambient=0.2,
            diffuse=0.8,
            specular=0.8,
            specular_power=40,
            smooth_shading=True,
        )

pl.background_color = "w"
pl.link_views()
pl.camera_position = "xy"
pos = pl.camera.position
pl.camera.position = (pos[0], pos[1] + 3, pos[2])
pl.camera.position = (10, 0, 0)
pl.camera.azimuth = 45
pl.camera.elevation = 30
light = pv.Light(
    position=(0, 0, 0), positional=True, cone_angle=50, exponent=20, intensity=0.2
)
pl.add_light(light)
pl.camera.zoom(0.7)
pl.show(screenshot="unet.png")
