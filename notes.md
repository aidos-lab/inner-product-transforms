# Comparison partners 


## Fast Point Cloud Generation with Straight Flows

- https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Fast_Point_Cloud_Generation_With_Straight_Flows_CVPR_2023_paper.pdf
- https://github.com/klightz/PSF
- Number of Params: 22M (Do double check) 
- Very fast inference times (0.04 sec)

**Summary**
- Work inspired by reformulation of diffusion models as 
    ODE's 
- Knowledge distillation used to map the diffusion model to one step 



3D Shape Generation and Completion through Point-Voxel Diffusion
- https://arxiv.org/pdf/2104.03670
- Unable to run code, non-compatible GPU


Deep Point Cloud Reconstruction 2021
- https://arxiv.org/abs/2111.11704
- No code avail able 
- Slides: https://iclr.cc/media/iclr-2022/Slides/6776_E28XvmT.pdf

Learning hierarchical composition for generative
modeling of set-structured Data 
- https://arxiv.org/pdf/2103.15619
- https://github.com/jw9730/setvae
- DO READ 
- Has parameter table. 
- Average is approx 2M parameters, minimum 750K parameters (set flow)


3D Point Cloud Geometry Compression on Deep Learnings
- https://april.zju.edu.cn/wp-content/papercite-data/pdf/huang20193dpc.pdf
- Might be of interest (compression of point clouds)

LION: Latent Point Diffusion Models for 3D Shape Generation
- https://proceedings.neurips.cc/paper_files/paper/2022/file/40e56dabe12095a5fc44a6e4c3835948-Paper-Conference.pdf


## Point-Voxel CNN for Efficient 3D Deep Learning
- https://arxiv.org/pdf/1907.03739

**Summary**
- Sensors, lidar detectors, view world as point cloud 
- Point cloud research: 
    - part segmentation in objects 
    - 3d segmentations in scene's
    - object detection in autonomous driving. 
- Deployment on small devices, such as cars 
    - efficient learning important 
- PC is very irregular + bigger inference cost and bigger 
- 



# Standard comparison partners 

1-GAN 
PC-GAN 



# Tasks 

Main question: 

What gap does our model fill in the market. 
- What tasks is our model good at? 
- Are those things that solve a problem?


Possible tasks: 
- Interpolation in latent space 
- Stability wrt noise. 
- Number of parameters 
- Compression of point clouds
-  


# Notes 

- A lot of recent work has been on either the transformer of diffusion side 
- So either slow inference from diffusion or quadratic memory from transformers. 
    - I think our model will never beat that, so pivot to a subtask might be required. 
- 
