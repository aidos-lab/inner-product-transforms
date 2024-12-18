"""
A wrapper to give our model the same signature as the 
one in pointflow and make it accept the same type of data.

"""

from scipy import sparse
import torch

from layers.directions import generate_uniform_directions
from layers.ect import EctLayer, compute_ect_point_cloud
from models.encoder import BaseLightningModel

DEVICE = "cuda:0"


def normalize(pts):
    assert pts.shape[1:] == (2048, 3)
    pts_means = pts.mean(axis=-2, keepdim=True)
    pts = pts - pts_means
    pts_norms = torch.norm(pts, dim=-1, keepdim=True).max(dim=-2, keepdim=True)[0]
    pts = pts / pts_norms
    return pts, pts_means, pts_norms


class ModelWrapper:
    def __init__(self, encoder, vae=None) -> None:
        self.encoder = encoder
        self.vae = vae
        if vae:
            self.vae.model.eval()

    @torch.no_grad()
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

    @torch.no_grad()
    def reconstruct(self, batch, normalized=False):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """
        pc_shape = batch[0].x.shape

        if normalized:
            batch.x, batch_means, batch_std = normalize(
                batch.x.view(-1, pc_shape[0], pc_shape[1])
            )

        if self.vae is not None:
            # Rescale to [-1,1] for the VAE
            ect = 2 * batch.ect - 1
            reconstructed_ect, _, _, _ = self.vae(ect.unsqueeze(1))
            # Rescale to 0,1
            reconstructed_ect = (reconstructed_ect + 1) / 2
        else:
            reconstructed_ect = batch.ect

        pointcloud = self.encoder(reconstructed_ect).view(
            -1, self.encoder.config.num_pts, pc_shape[1]
        )

        if normalized:
            pointcloud = pointcloud * batch_std + batch_means

        return pointcloud


class ModelDownsampleWrapper:
    def __init__(
        self,
        encoder_downsampler: BaseLightningModel,
        encoder_upsampler: BaseLightningModel,
    ) -> None:
        self.encoder_downsampler = encoder_downsampler.eval()
        self.encoder_upsampler = encoder_upsampler.eval()
        self.config = encoder_downsampler.config
        self.v = generate_uniform_directions(
            num_thetas=self.config.ectconfig.num_thetas,
            d=self.config.ectconfig.ambient_dimension,
            seed=self.config.ectconfig.seed,
        ).cuda()

    @torch.no_grad()
    def reconstruct(self, batch, normalized=False):
        """
        The method first applies the downsamler to create a sparse
        representation of the point cloud and subsequently upsamples the model
        using the ECT of the sparse representation. The upsampling model has
        _not_ been finetunes and consists of the standard encoder. The ECT's it
        is decoding in this experiment are thus far out of distribution.
        """

        pc_shape = batch[0].x.shape

        sparse_pointcloud = self.encoder_downsampler(batch.ect).view(
            -1, self.config.num_pts, pc_shape[1]
        )

        sparse_ect = (
            compute_ect_point_cloud(
                x=sparse_pointcloud,
                v=self.v,
                radius=self.config.ectconfig.r,
                resolution=self.config.ectconfig.resolution,
                scale=self.config.ectconfig.scale,
            )
            / self.config.num_pts
        )
        pointcloud = self.encoder_upsampler(sparse_ect).view(
            -1, self.encoder_upsampler.config.num_pts, pc_shape[1]
        )

        return pointcloud
