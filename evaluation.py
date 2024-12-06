import torch

from shapesynthesis.metrics.PyTorchEMD.emd_cuda import (
    earth_mover_distance as EMD,
)


if __name__ == "__main__":
    B, N = 2, 10
    x = torch.rand(B, N, 3)
    y = torch.rand(B, N, 3)

    emd_batch = EMD(x.cuda(), y.cuda(), False)
    print(emd_batch.shape)
    print(emd_batch.mean().detach().item())
