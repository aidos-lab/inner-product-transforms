import time
import statistics

import torch

from load_models import load_encoder, load_vae

torch.set_float32_matmul_precision("high")

DEVICE = "cpu"
NUM_ITER = 10000


encoder_model = load_encoder("./trained_models/ectencoder_shapenet_airplane.ckpt")
vae = load_vae("./trained_models/vae_shapenet_airplane.ckpt")
vae.model.eval().to(DEVICE)
encoder_model.eval().to(DEVICE)

print(vae.device)
print(encoder_model.device)
print(vae.layer.lin)

# encoder_model = torch.compile(encoder_model)
# vae = torch.compile(vae)

# Warm-up runs
x = torch.rand(size=(2, 1, 96, 96)).to(DEVICE)
for _ in range(10):
    with torch.no_grad():
        _ = encoder_model(x)
# Multiple iterations
inference_times = []
for _ in range(NUM_ITER):
    with torch.no_grad():
        start_time = time.time()
        _ = encoder_model(x)
        end_time = time.time()
    inference_times.append(end_time - start_time)
average_inference_time = statistics.mean(inference_times)
print(f"Average Encoder inference time: {average_inference_time:.8f} seconds")

# # Warm-up runs
# x = torch.rand(size=(2, 1, 96, 96)).to(DEVICE)
# for _ in range(10):
#     recon, _, _, _ = vae.model(x)
#     res = encoder_model((recon + 1) / 2)
#
# # Multiple iterations
# inference_times = []
# for _ in range(NUM_ITER):
#     with torch.no_grad():
#         start_time = time.time()
#         recon, _, _, _ = vae.model(x)
#         res = encoder_model((recon + 1) / 2)
#         end_time = time.time()
#     inference_times.append(end_time - start_time)
# average_inference_time = statistics.mean(inference_times)
# print(f"Average AutoEncoder inference time: {average_inference_time:.8f} seconds")
#
# # %%
# # Warm-up runs
# for _ in range(10):
#     samples = vae.model.sample(10, DEVICE)
#     res = encoder_model(samples)
# # Multiple iterations
# inference_times = []
# for _ in range(NUM_ITER):
#     with torch.no_grad():
#         start_time = time.time()
#         samples = vae.model.sample(1, DEVICE)
#         res = encoder_model((samples + 1) / 2)
#         end_time = time.time()
#     inference_times.append(end_time - start_time)
# average_inference_time = statistics.mean(inference_times)
# print(f"Average Sampling inference time: {average_inference_time:.8f} seconds")
