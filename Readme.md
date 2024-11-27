
```{note}
This is a note
```

# Topological Reconstruction and Interpolation of Graphs and Point Clouds

The Euler Characteristic Transform is an expressive and fast to compute.
However inversion of Euler Characteristic Transforms has hitherto only focussed
on specialized cases, such as grayscale images and theoretical treatises for
graphs in 2D.

We propose a machine learning based approach for the inversion of ECT's of
shapes in two and three dimensions. Using a structured approach in choosing the
directions, we find that machine learning algorithms can capture the essence of
shapes through the ECT.

By training a Variational Autoencoder to reconstruct the Euler Characteristic
transform to a $64$ dimensional latent space. This construction allows us to
compress a pointcloud of $1024$ points in 3D to a $64$ dimensional latent
vector and reconstruct the original point cloud from it. In the top row of the
ECT's we find the original ECT's and below it the reconstructed ECT's using our
VAE.

<!--
![Reconstructed-Pointcloud-ECT](figures/reconstructed_modelnet/reconstructed_ect.png)
-->

We then pass each ECT through our decoder model obtain the following
reconstructed point clouds.

<!--
![Reconstructed-Pointcloud](figures/reconstructed_modelnet/orbit_cloud.gif) -->

Moreover, we are able to _sample_ from this latent space to obtain novel
_generated_ Euler Characteristic transforms which in turn can be decoded using
our trained decoding model.

<!--
![Reconstructed-Pointcloud-ECT](figures/generated_modelnet/generated_ect.png)
-->

<!-- ![Reconstructed-Pointcloud](figures/generated_modelnet/orbit_cloud.gif)
-->

# MNIST

## Reconstruction results
Results for the MNIST dataset. 

First we reconstruct the numbers 0-9 from the ect to a point cloud. 
The top row is the reconstruction and the bottom row the input ect. 

![Reconstructed-Pointcloud](figures/img/mnist/reconstructed_all.png)

Next we test how well the Variational Auto Encoder is able to reconstruct the 
ECT. The top row represents the original ECT (same as bottom row above) and the 
bottom row is the reconstructed ECT.

![Reconstructed-Pointcloud](figures/img/mnist/reconstructed_ect_vae.png)

Once we are confident in the VAE's ability to faithfully reconstruct ect's, we 
need to verify that the reconstructed ect also provides good input for the model 
prediction since they got trained separately. 
The top row is the original data, the second row the reconstruction from the 
encoder model and third row the reconstruction of the point with the ect passed 
through vae first. 
The blurring, native to VAE's, causes for instance the 0 and 2 to be confused. 

![Reconstructed-Pointcloud](figures/img/mnist/reconstructed_pointcloud_vae.png)

The VAE also allow us to sample from the 64 dimensional latent space and obtain 
generated ECT's. These generated ect's then get passed through our decoder model 
to finally obtain plausable reconstructions of point clouds. 


![Reconstructed-Pointcloud](figures/img/mnist/generated_samples_vae.png)

## Emperical stability
### Noisy pointclouds
We check how stable the encoder model is with respect to adding normally distributed
noise to the point cloud. In the top left we find the original ground truth with 
the corresponding ect in the bottom row. In the middle row we have the reconstruction 
result for the computed ect. 




![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_0.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_1.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_2.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_3.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_4.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_5.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_6.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_7.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_8.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ambient_noise_9.png)


### Noisy ECT
In this experiment we test how stable the ect is with respect to normally distributed 
noise in the ect domain.
From left to right we repeatedly add normally distributed noise with std of .05.
Then we pass the noisy ect through our decoder plot both the ect and the 
predicted point clouds. 
The top row are the reconstructed pointcloud corresponding to the bottom noisy 
ect. The left most ect has no noise. 

![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_0.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_1.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_2.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_3.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_4.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_5.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_6.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_7.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_8.png)
![Reconstructed-Pointcloud](figures/img/mnist/stability_ect_noise_9.png)


## Modelnet
### Reconstructed modelnet
![Reconstructed-Pointcloud](figures/img/modelnet/orbit_cloud.gif)

![Reconstructed-Pointcloud](figures/img/modelnet/reconstructed_pointcloud.png)

### Reconstructed vae modelnet
![Reconstructed-Pointcloud](figures/img/modelnet/reconstructed_ect.png)

![Reconstructed-Pointcloud](figures/img/modelnet/reconstructed_vae_pointcloud.png)


### Generated vae modelnet
![Reconstructed-Pointcloud](figures/img/modelnet/generated_ect.png)
![Reconstructed-Pointcloud](figures/img/modelnet/generated_pointcloud.png)
![Reconstructed-Pointcloud](figures/img/modelnet/generated_pointcloud.gif)



## DSprites

### Reconstructed Dsprites
![Reconstructed-Pointcloud](figures/img/dsprites/reconstructed_all.png)


### Reconstructed VAE

![Reconstructed-Pointcloud](figures/img/dsprites/reconstructed_ect_vae.png)
![Reconstructed-Pointcloud](figures/img/dsprites/reconstructed_pointcloud_vae.png)

### Reconstructed Dsprites
![Reconstructed-Pointcloud](figures/img/dsprites/generated_samples_vae.png)



## Topological 
### Reconstructed pointclouds
![Reconstructed-Pointcloud](figures/img/topological/reconstructed_pointcloud.gif)

### Reconstructed VAE Pointcloud
![Reconstructed-Pointcloud](figures/img/topological/reconstructed_vae_pointcloud.png)

### Generated ECT & Pointcloud
![Reconstructed-Pointcloud](figures/img/topological/generated_ect.png)
![Reconstructed-Pointcloud](figures/img/topological/generated_pointcloud.png)


# Reproducibility

The models used to generate the results and tables in the our work are released 
as pytorch lightning models in the release section of the repository.
