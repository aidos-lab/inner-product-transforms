import torch
import torch.nn as nn

def compute_mse_loss_fn(decoded,z_mean,z_log_var, ect):
        kl_div = -0.5 * torch.sum(
        1 + z_log_var - z_mean**2 - torch.exp(z_log_var),
        axis=1,
        )  # sum over latent dimension
        batchsize = kl_div.size(0)
        kl_div = kl_div.mean()  # average over batch dimension

        pixelwise = nn.functional.mse_loss(decoded, ect, reduction="none")
        pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
        pixelwise = pixelwise.mean()  # average over batch dimension

        return pixelwise + kl_div

