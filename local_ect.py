import torch 
from torch_geometric.data import Data, Batch 

x = torch.tensor([[1,1],[2,2],[3,3]])
batch = Batch.from_data_list([Data(x=x),Data(x=x+3)])

_batch = batch.clone()
x_hat = _batch.x.view(-1,3,2).repeat_interleave(3,dim=0)
_batch.x = (x_hat - _batch.x.unsqueeze(1)).view(-1,2)
_batch.batch = _batch.batch.repeat_interleave(3,dim=0)




print(x_centered)
print(batch.batch.repeat_interleave(3,dim=0))
