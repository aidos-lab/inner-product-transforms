from shapesynthesis.loaders import load_config, load_model

config, _ = load_config("../configs/encoder_128_airplane.yaml")

# config
model = load_model(config.modelconfig, "../trained_models/encoder_128_airplane.ckpt")

# |%%--%%| <UAvmrWYDlR|vXfG6UUzht>

import matplotlib.pyplot as plt
import torch

ects = torch.load("../generated_ects_full.pt")


print(ects[0].min(), ects[0].max(), ects.shape)

plt.imshow(ects[0].squeeze())

print(ects[0])


# |%%--%%| <vXfG6UUzht|BZ3E3HIGgC>


# |%%--%%| <BZ3E3HIGgC|nbriU8y3FP>

with torch.no_grad():
    pts = model.model(ects).view(-1, 2048, 3).cpu()

pts.shape

# |%%--%%| <nbriU8y3FP|Bvn81wh9TZ>

import pyvista as pv
import torch

pv.set_jupyter_backend("static")


def plot_grid(pts, name="grid.png"):
    pl = pv.Plotter(
        shape=(8, 8),
        window_size=[1600, 1600],
        border=False,
        polygon_smoothing=True,
    )

    for col in range(8):
        for row in range(8):
            # First plat
            pl.subplot(row, col)
            actor = pl.add_points(
                pts[col * 8 + row].cpu().numpy().reshape(-1, 3),
                style="points",
                emissive=False,
                show_scalar_bar=False,
                render_points_as_spheres=True,
                color="lightgray",
                point_size=2,
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
    pl.camera.position = (5, 0, 0)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    # create a top down light
    light = pv.Light(
        position=(0, 0, 0), positional=True, cone_angle=50, exponent=20, intensity=0.2
    )
    pl.add_light(light)
    pl.camera.zoom(3.0)
    pl.show(screenshot=True)
    pl.screenshot(name)


# |%%--%%| <Bvn81wh9TZ|aPY3wQGObv>

plot_grid(0.1 * pts)

# |%%--%%| <aPY3wQGObv|2X010h72V3>

from shapesynthesis.metrics.evaluation import compute_all_metrics

airplane_refs = torch.load("../results/vae_airplane_latent/references.pt").cpu()
s = torch.load("../results/vae_airplane_latent/stdevs.pt").cpu()
m = torch.load("../results/vae_airplane_latent/means.pt").cpu()

pts = pts[:405] * s + m

print(s.shape, m.shape)


# |%%--%%| <2X010h72V3|MeV5t11LBd>

from plotting import plot_recon_3d

plot_recon_3d(airplane_refs.cpu().numpy() * 10, pts.numpy() * 10, num_pc=13)

# |%%--%%| <MeV5t11LBd|ePtOujtXFa>

compute_all_metrics(pts, airplane_refs, batch_size=100)
