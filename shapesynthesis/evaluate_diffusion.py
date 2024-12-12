import os
from typing import List, Optional, Union
import torch
import lightning as L
from models.diffusion import BaseModel
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt

lightning_model = BaseModel.load_from_checkpoint(
    "./trained_models/diffusion_mnist.ckpt"
)

model = lightning_model.model
noise_scheduler = lightning_model.noise_scheduler


batch_size = 1
generator = torch.Generator(device="cpu").manual_seed(1234)
num_inference_steps = 1000
output_type = "pil"

with torch.no_grad():
    image_shape = (batch_size, 64, 128)
    image = torch.rand(image_shape, generator=generator)  # .cuda()

    # set step values
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in range(num_inference_steps):
        # 1. predict noise model_output
        model_output = model(image, t).sample

        # 2. compute previous image: x_t -> x_t-1
        image = noise_scheduler.step(
            model_output, t, image, generator=generator
        ).prev_sample


image.cpu().detach()
plt.imshow(image.cpu().detach().squeeze())
plt.show()
# # After each epoch you optionally sample some demo images with evaluate() and save the model
# pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

# # Sample some images from random noise (this is the backward diffusion process).
# # The default pipeline output type is `List[PIL.Image]`
# images = pipeline(
#     batch_size=32,
#     generator=torch.Generator(device="cpu").manual_seed(
#         1234
#     ),  # Use a separate torch generator to avoid rewinding the random state of the main training loop
# ).images

# print(images)


# # Make a grid out of the images
# image_grid = make_image_grid(images, rows=4, cols=4)

# # Save the images
# test_dir = os.path.join(config.output_dir, "samples")
# os.makedirs(test_dir, exist_ok=True)
# image_grid.save(f"{test_dir}/{epoch:04d}.png")
