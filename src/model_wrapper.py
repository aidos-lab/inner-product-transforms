"""
A wrapper to give our model the same signature as the
one in pointflow and make it accept the same type of data.
"""

import torch
from layers.directions import generate_uniform_directions
from layers.ect import EctLayer, compute_ect_point_cloud
from models.encoder_new import BaseLightningModel as Encoder
from models.vae_baseline import BaseLightningModel as VAE
from scipy import sparse

DEVICE = "cuda:0"


def normalize(pts):
    assert pts.shape[1:] == (2048, 3)
    pts_means = pts.mean(axis=-2, keepdim=True)
    pts = pts - pts_means
    pts_norms = torch.norm(pts, dim=-1, keepdim=True).max(dim=-2, keepdim=True)[0]
    pts = pts / pts_norms
    return pts, pts_means, pts_norms


class ModelWrapper:
    def __init__(self, encoder: Encoder, vae: VAE | None = None) -> None:
        self.encoder = encoder
        self.encoder.eval()
        self.vae = vae
        if vae is not None:
            self.vae.model.eval()

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        out_pc, sample_ect = model.sample(len(batch), pc_shape)
        The way we expect the input
        B is the number of point clouds
        N is the number of points per cloud.
        _, out_pc = model.sample(B, N)
        """
        ect_samples = self.vae.model.sample(n=num_samples)

        vae_pointcloud = self.encoder(ect_samples).view(
            -1,
            self.encoder.config.num_pts,
            3,
        )
        return vae_pointcloud, ect_samples

    @torch.no_grad()
    def reconstruct(self, ect):
        if self.vae is not None:
            output = self.vae.model(ect)
            pointcloud = self.encoder(output[0].squeeze()).view(
                -1, self.encoder.config.num_pts, 3
            )
            reconstructed_ect = output[0]
        else:
            pointcloud = self.encoder(ect.squeeze()).view(
                -1, self.encoder.config.num_pts, 3
            )
            reconstructed_ect = ect

        return pointcloud, reconstructed_ect


class ModelNoOpWrapper:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def sample(self, num_samples: int, num_points: int, ambient_dimension: int):
        pass

    @torch.no_grad()
    def reconstruct(self, batch, normalized=False):
        """
        Takes in a pointcloud of the form BxPxD
        and does nothing.
        """
        pc_shape = batch[0].x.shape

        pointcloud = batch.x.view(-1, pc_shape[0], pc_shape[1])

        return pointcloud, batch.ect


class ModelDownsampleWrapper:
    def __init__(
        self,
        encoder_downsampler: Encoder,
        encoder_upsampler: Encoder,
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
    def reconstruct(self, ect_gt, normalized=False):
        """
        The method first applies the downsamler to create a sparse
        representation of the point cloud and subsequently upsamples the model
        using the ECT of the sparse representation. The upsampling model has
        _not_ been finetunes and consists of the standard encoder. The ECT's it
        is decoding in this experiment are thus far out of distribution.
        """

        sparse_pointcloud = self.encoder_downsampler(ect_gt).view(
            -1,
            self.config.num_pts,
            self.config.ectconfig.ambient_dimension,
        )

        sparse_ect = self.encoder_downsampler.ect_transform(sparse_pointcloud)
        pointcloud = self.encoder_upsampler(sparse_ect).view(
            -1, self.encoder_upsampler.config.num_pts, 3
        )

        return pointcloud, sparse_pointcloud, sparse_ect


class ModelCompletionWrapper:
    def __init__(
        self,
        encoder: Encoder,
    ) -> None:
        self.encoder = encoder.eval()
        self.config = encoder.config

        # Sample 200 points
        self.subset_indexes = torch.randperm(n=self.config.num_pts)[:256]
        self.v = generate_uniform_directions(
            num_thetas=self.config.ectconfig.num_thetas,
            d=self.config.ectconfig.ambient_dimension,
            seed=self.config.ectconfig.seed,
        ).cuda()

    @torch.no_grad()
    def reconstruct(self, batch, normalized=False):
        """
        The method first subsamples the point cloud to create a point cloud that
        it can recreate. The completion model has _not_ been finetuned and
        consists of the standard encoder. The ECT's it is decoding in this
        experiment are thus far out of distribution.
        """

        pc_shape = batch[0].x.shape

        print(batch.x.shape)
        # Select 200 random point from the dataset.
        sparse_pointcloud = batch.x.view(-1, 2048, 3)[:, self.subset_indexes, :]

        print("computeing ect")
        print(sparse_pointcloud.shape)
        sparse_ect = (
            compute_ect_point_cloud(
                x=sparse_pointcloud,
                v=self.v,
                radius=self.config.ectconfig.r,
                resolution=self.config.ectconfig.resolution,
                scale=self.config.ectconfig.scale,
            )
            / 200
        )

        # Complete the point cloud to 2048 points.
        pointcloud = self.encoder(sparse_ect).view(
            -1, self.encoder.config.num_pts, pc_shape[1]
        )

        return pointcloud, sparse_pointcloud
