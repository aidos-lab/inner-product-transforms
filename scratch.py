import torch

from kaolin.metrics.pointcloud import chamfer_distance


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


a = torch.rand(size=(3, 100, 2))
b = torch.rand(size=(3, 100, 2))

dl, dr = distChamfer(a, b)
print(dl.mean(dim=1) + dr.mean(dim=1))

# cd = chamfer_distance(a.cuda(), b.cuda())
# print(cd)
