import torch 
import numpy 

from kaolin.metrics.pointcloud import chamfer_distance
torch.set_default_dtype(torch.float16)

a = torch.rand(size=(10,10,3)).cuda()
b = torch.rand(size=(10,10,3)).cuda()

print(chamfer_distance(a,b))
print(a.dtype)
print(b.dtype)


# import torch.nn.functional as F

# a = torch.normal(mean=0,std=1,size=(100,))
# b = torch.normal(mean=0,std=1,size=(100,))
# out = F.kl_div(a, b,reduction="batchmean")
#

#
# # # this is the same example in wiki
# # P = torch.Tensor([0.36, 0.48, 0.16])
# # Q = torch.Tensor([0.333, 0.333, 0.333])
#
#
# P = torch.normal(mean=0,std=1,size=(100,))
# Q = torch.normal(mean=3,std=1,size=(100,))
#
#
# r1 = (P * (P / Q).log()).sum()
# # tensor(0.0863), 10.2 µs ± 508
#
# r2 = F.kl_div(Q.log(), P, None, None, 'sum')
# # tensor(0.0863), 14.1 µs ± 408 ns
#
# print(r1)
# print(r2)
