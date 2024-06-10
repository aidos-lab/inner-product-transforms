# Topological Reconstruction and Interpolation of Graphs and Point Clouds

The Euler Characteristic Transform is an expressive and fast to compute. However inversion of Euler Characteristic Transforms has hitherto only focussed on specialized cases, such as grayscale images and theoretical treatises for graphs in 2D.

We propose a machine learning based approach for the inversion of ECT's of shapes in two and three dimensions. Using a structured approach in choosing the directions, we find that machine learning algorithms can capture the essence of shapes through the ECT.

By training a Variational Autoencoder to reconstruct the Euler Characteristic transform to a $64$ dimensional latent space. This construction allows us to compress a pointcloud of $1024$ points in 3D to a $64$ dimensional latent vector and reconstruct the original point cloud from it. In the top row of the ECT's we find the original ECT's and below it the reconstructed ECT's using our VAE.

![Reconstructed-Pointcloud-ECT](figures/reconstructed_modelnet/reconstructed_ect.png)

We then pass each ECT through our decoder model obtain the following reconstructed point clouds.

![Reconstructed-Pointcloud](figures/reconstructed_modelnet/orbit_cloud.gif)

Moreover, we are able to _sample_ from this latent space to obtain novel _generated_ Euler Characteristic transforms which in turn can be decoded using our trained decoding model.

![Reconstructed-Pointcloud-ECT](figures/generated_modelnet/generated_ect.png)

![Reconstructed-Pointcloud](figures/generated_modelnet/orbit_cloud.gif)

# Reproducibility

The models used to generate the results and tables in the our work are released as pytorch lightning models in the release section of the repository.
