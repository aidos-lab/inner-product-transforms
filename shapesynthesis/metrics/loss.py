import torch
import torch.nn.functional as F
from torchmetrics.regression import KLDivergence
from kaolin.metrics.pointcloud import chamfer_distance


def compute_mse_loss_fn(ect_hat, ect):
    pixelwise = F.mse_loss(ect_hat, ect)
    return pixelwise


# I am not sure if I used mean or sum here.
def compute_mse_kld_loss_fn(decoded, mu, log_var, ect, beta=0.0):
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )

    mse_loss = F.mse_loss(decoded, ect, reduction="mean")

    return mse_loss + beta * kld_loss, kld_loss, mse_loss


def chamfer3D(pred_pc, ref_pc):
    return chamfer_distance(pred_pc, ref_pc).mean()


def chamfer2D(pred_pc, ref_pc):
    pred_pc = F.pad(input=pred_pc, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)
    ref_pc = F.pad(input=ref_pc, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)
    return chamfer_distance(pred_pc, ref_pc).mean()


def chamfer3DECT(pred_pc, ref_pc, ect_pred, ect):
    ch_loss = chamfer3D(pred_pc, ref_pc)
    mse_loss = F.mse_loss(ect_pred, ect)
    return ch_loss + 10 * mse_loss, mse_loss, ch_loss


def chamfer2DECT(pred_pc, ref_pc, ect_pred, ect):
    ch_loss = chamfer2D(pred_pc, ref_pc)
    mse_loss = F.mse_loss(ect_pred, ect)
    return ch_loss + 10 * mse_loss, mse_loss, ch_loss


def dect_loss_fn(ect_pred, ect):
    mse_loss = F.mse_loss(ect_pred, ect)
    return mse_loss
