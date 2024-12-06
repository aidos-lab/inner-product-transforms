"""
Gradient ablation with respect to the number of directions 
and the resolution. 

It follows the point cloud optimization experiment. 
In given a point at x=-0.5 and a point at x_hat=.5 
we calculate the gradient between the "ect" of both. 
This can be done analytically, and we aim to 
understand the comparative difference between the true
gradient and discretized version for various scale 
values and various resolutions. 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

resolutions = []
for resolution in range(2, 512):
    gradients = []
    for scale in range(1, 128):
        # Set x values, remains fixed.
        x = torch.tensor(0.2, requires_grad=True)
        x_hat = torch.tensor(0.4)

        lin = torch.tensor(np.linspace(-1, 1, resolution))

        ect = torch.nn.functional.sigmoid(scale * (lin - x))
        ect_hat = torch.nn.functional.sigmoid(scale * (lin - x_hat))

        loss = torch.nn.functional.mse_loss(ect, ect_hat)
        loss.backward()
        gradients.append(x.grad)
    resolutions.append(gradients)

plt.imshow(torch.tensor(resolutions))
plt.show()
