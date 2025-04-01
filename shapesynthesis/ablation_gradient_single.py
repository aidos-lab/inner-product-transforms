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

# Hyper parameters
radius = 4.0
resolution = 16
scale = 4

gradients = []
for resolution in range(1, 64):
    # Set x values, remains fixed.
    x = torch.tensor(-0.4 * radius, requires_grad=True)
    x_hat = torch.tensor(0.4 * radius)

    lin = torch.tensor(np.linspace(-radius, radius, resolution))

    ect = torch.nn.functional.sigmoid(scale * (lin - x))
    ect_hat = torch.nn.functional.sigmoid(scale * (lin - x_hat))

    loss = torch.nn.functional.mse_loss(ect, ect_hat)
    loss.backward()
    gradients.append(x.grad.clone())


plt.plot(torch.stack(gradients))
plt.show()

"""
Conclusion:
The resolution needs to be larger than the scale,
then the gradients are stable *and* constant wrt
the resolution.

And the scale needs to be larger than min threshold. 

"""
