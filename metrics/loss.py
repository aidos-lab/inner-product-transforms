import torch
import torch.nn.functional as F

# recons = args[0]
# input = args[1]
# mu = args[2]
# log_var = args[3]

# kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
# recons_loss = F.mse_loss(recons, input)

# kld_loss = torch.mean(
#     -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
# )

# loss = recons_loss + kld_weight * kld_loss
# return {
#     "loss": loss,
#     "Reconstruction_Loss": recons_loss.detach(),
#     "KLD": -kld_loss.detach(),
# }


def compute_mse_loss_fn(ect_hat, ect):
    pixelwise = F.mse_loss(ect_hat, ect)
    return pixelwise


def compute_mse_kld_loss_fn(decoded, mu, log_var, ect):
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )

    pixelwise = F.mse_loss(decoded, ect)

    return pixelwise + 0.00025 * kld_loss
