import time
import statistics

import torch

from load_models import load_encoder, load_vae
from torch_geometric.data import Data, Batch

torch.set_float32_matmul_precision("high")

DEVICE = "cuda:0"
NUM_ITER = 100


encoder_model = load_encoder("./trained_models/ectencoder_shapenet_airplane.ckpt")
vae = load_vae("./trained_models/vae_shapenet_airplane.ckpt")
vae.model.eval().to(DEVICE)
encoder_model.eval().to(DEVICE)
encoder_model.layer.v = encoder_model.layer.v.to(DEVICE)
encoder_model.layer.lin = encoder_model.layer.lin.to(DEVICE)
# print(vae.device)
print(encoder_model.device)
# print(encoder_model.device)
# print(vae.layer.lin)

# encoder_model = torch.compile(encoder_model)
# vae = torch.compile(vae)

# # Warm-up runs
# x = torch.rand(size=(2, 1, 96, 96)).to(DEVICE)
# for _ in range(10):
#     with torch.no_grad():
#         _ = encoder_model(x)
# # Multiple iterations
# inference_times = []
# for _ in range(NUM_ITER):
#     with torch.no_grad():
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         _ = encoder_model(x)
#         end.record()
#         torch.cuda.synchronize()
#     inference_times.append(start.elapsed_time(end) / 1000)
# average_inference_time = statistics.mean(inference_times)
# print(f"Average Encoder inference time: {average_inference_time:.8f} seconds")

# # Warm-up runs
# x = torch.rand(size=(2, 1, 96, 96)).to(DEVICE)
# for _ in range(10):
#     recon, _, _, _ = vae.model(x)
#     res = encoder_model((recon + 1) / 2)

# # Multiple iterations
# inference_times = []
# for _ in range(NUM_ITER):
#     with torch.no_grad():
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         recon, _, _, _ = vae.model(x)
#         res = encoder_model((recon + 1) / 2)
#         end.record()
#         torch.cuda.synchronize()

#         end_time = time.time()
#     inference_times.append(start.elapsed_time(end) / 1000)
# average_inference_time = statistics.mean(inference_times)
# print(f"Average AutoEncoder inference time: {average_inference_time:.8f} seconds")

# # Warm-up runs
# for _ in range(10):
#     samples = vae.model.sample(10, DEVICE)
#     res = encoder_model(samples)
# # Multiple iterations
# inference_times = []
# for _ in range(NUM_ITER):
#     with torch.no_grad():
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         samples = vae.model.sample(1, DEVICE)
#         res = encoder_model((samples + 1) / 2)
#         end.record()
#         torch.cuda.synchronize()

#         end_time = time.time()
#     inference_times.append(start.elapsed_time(end) / 1000)
# average_inference_time = statistics.mean(inference_times)
# print(f"Average Sampling inference time: {average_inference_time:.8f} seconds")


print("===== Including ECT Layer ====")


# Warm-up runs
batch = Batch.from_data_list([Data(x=torch.rand(size=(2048, 3)))]).to(DEVICE)

for _ in range(10):
    with torch.no_grad():
        ect = encoder_model.layer(batch, batch.batch)
        _ = encoder_model(ect)
# Multiple iterations
inference_times = []
for _ in range(NUM_ITER):
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ect = encoder_model.layer(batch, batch.batch)
        _ = encoder_model(ect)
        end.record()
        torch.cuda.synchronize()
    inference_times.append(start.elapsed_time(end) / 1000)
average_inference_time = statistics.mean(inference_times)
print(f"Average Encoder inference time: {average_inference_time:.8f} seconds")

# Warm-up runs
batch = Batch.from_data_list([Data(x=torch.rand(size=(2048, 3)))]).to(DEVICE)
for _ in range(10):
    ect = encoder_model.layer(batch, batch.batch)
    recon, _, _, _ = vae.model(ect.unsqueeze(0))
    res = encoder_model((recon + 1) / 2)

# Multiple iterations
inference_times = []
for _ in range(NUM_ITER):
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ect = encoder_model.layer(batch, batch.batch)
        recon, _, _, _ = vae.model(ect.unsqueeze(0))
        res = encoder_model((recon + 1) / 2)
        end.record()
        torch.cuda.synchronize()

        end_time = time.time()
    inference_times.append(start.elapsed_time(end) / 1000)
average_inference_time = statistics.mean(inference_times)
print(f"Average AutoEncoder inference time: {average_inference_time:.8f} seconds")
