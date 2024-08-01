"""
A wrapper to give our model the same signature as the 
one in pointflow and make it accept the same type of data.

"""

from torch import nn
from load_models import load_encoder, load_vae
from torch_geometric.data import Data, Batch


class TopologicalModel(nn.Module):
    def __init__(self, encoder_config, vae_config) -> None:
        self.encoder = load_encoder(encoder_config)
        self.vae = load_vae(vae_config)

    def reconstruct(self, x):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """

        # Reshape into a torch_geometric batch

        batch = Batch.from_data_list([Data(x=pts.view(-1, 3)) for pts in x])

        # Compute the ECT
        ect = self.encoder.layer(batch)

        # Compute the VAE reconstruction

        # reconstructed_ect, _,_,_

        # Compute the ECT reconstruction
