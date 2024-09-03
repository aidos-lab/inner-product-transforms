"""wr
A wrapper to give our model the same signature as the 
one in pointflow and make it accept the same type of data.

"""

from json import encoder
from torch import inference_mode, nn
from load_models import load_encoder, load_vae
from torch_geometric.data import Data, Batch
import torch

DEVICE = "cuda:0"


class TopologicalModelVAE:
    def __init__(self, encoder, vae) -> None:
        self.encoder = encoder
        self.vae = vae
        # self.vae.model.eval()

    def sample(self, B, N):
        """
        The way we expect the input
        B is the number of point clouds
        N is the number of points per cloud.
        _, out_pc = model.sample(B, N)
        """
        ect_samples = self.vae.model.sample(B, "cuda:0")

        # Rescale to 0,1
        ect_samples = (ect_samples + 1) / 2

        vae_pointcloud = self.encoder(ect_samples).view(B, N, 3)
        return None, vae_pointcloud

    def reconstruct(self, x, num_points=2048):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """

        x_means = x.mean(axis=1, keepdim=True)
        x = x - x_means

        x_norms = torch.norm(x, dim=2).max(axis=1)[0].reshape(-1, 1, 1)
        x = x / x_norms

        # Reshape into a torch_geometric batch

        batch = Batch.from_data_list([Data(x=pts.view(-1, 3)) for pts in x])

        batch = batch.to(DEVICE)
        ect = self.encoder.layer(batch, batch.batch)
        # encoder_pointcloud = self.encoder_model(ect)
        ect = 2 * ect - 1
        reconstructed_ect, _, _, _ = self.vae(ect.unsqueeze(1))

        # Rescale to 0,1
        reconstructed_ect = (reconstructed_ect + 1) / 2

        vae_pointcloud = self.encoder(reconstructed_ect).view(-1, num_points, 3)

        vae_pointcloud = vae_pointcloud * x_norms
        vae_pointcloud = vae_pointcloud + x_means

        assert vae_pointcloud.shape == x.shape

        return vae_pointcloud


class TopologicalModelEncoder:
    def __init__(self, encoder_model) -> None:
        super().__init__()
        self.encoder_model = encoder_model

    def reconstruct(self, x, num_points=2048):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """

        x_means = torch.mean(x, axis=-2)
        x = x - x_means.unsqueeze(1)
        # print("-------POINT CLOUD MEANS ======")
        # print(x_means)

        x_norms = torch.norm(x, dim=-1).max(axis=1)[0].reshape(-1, 1, 1)
        x = x / x_norms

        # Reshape into a torch_geometric batch

        batch = Batch.from_data_list([Data(x=pts.view(-1, 3)) for pts in x])

        batch = batch.to(DEVICE)
        ect = self.encoder_model.layer(batch, batch.batch)
        encoder_pointcloud = self.encoder_model(ect).view(-1, num_points, 3)

        # print("-------DECODED POINT CLOUD BEFORE NORMALIZATION ======")
        # print(encoder_pointcloud.mean(dim=-2))
        # print("-------======")

        encoder_pointcloud = encoder_pointcloud * x_norms
        encoder_pointcloud = encoder_pointcloud + x_means.unsqueeze(1)

        # print("-------DECODED AFTER NORMALIZATION ======")
        # print(encoder_pointcloud.mean(dim=-2))
        # print("---------------------------------------------")
        # raise "bye"

        assert encoder_pointcloud.shape == x.shape

        return encoder_pointcloud


class TopologicalModelEncoderScaled:
    def __init__(self, encoder_model) -> None:
        super().__init__()
        self.encoder_model = encoder_model

    def reconstruct(self, x, num_points=2048):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """

        # Reshape into a torch_geometric batch

        batch = Batch.from_data_list([Data(x=pts.view(-1, 3)) for pts in x])

        batch = batch.to(DEVICE)
        ect = self.encoder_model.layer(batch, batch.batch)
        encoder_pointcloud = self.encoder_model(ect).view(-1, num_points, 3)

        assert encoder_pointcloud.shape == x.shape

        return encoder_pointcloud


if __name__ == "__main__":
    from omegaconf import OmegaConf

    ect_config = OmegaConf.load("./configs/config_encoder_shapenet.yaml")
    vae_config = OmegaConf.load("./configs/config_vae_shapenet.yaml")

    encoder_model = load_encoder(ect_config)
    vae = load_vae(vae_config)
    model = TopologicalModelVAE(encoder_model, vae)


