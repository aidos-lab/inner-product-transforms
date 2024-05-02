import torch
import torch.nn as nn
from layers.config import EctConfig
from torch_geometric.data import Batch, Data


def compute_wecc(nh, index, lin, weight,out):
    ecc = torch.nn.functional.sigmoid(500 * torch.sub(lin, nh)) * weight.view(1,-1,1)
    res = torch.index_add(out,1, index, ecc).movedim(0, 1)
    return res


def compute_wect(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    # fh, _ = nh[data.face].min(dim=0)

    # Compute the weights 
    node_weights = torch.ones(nh.shape[0])
    edge_weights = data.edge_weights.view(-1,1)

    # raise "hello"
    return (
        compute_wecc(nh, data.batch, lin, node_weights,out)
        - compute_wecc(eh, data.batch[data.edge_index[0]], lin, edge_weights,out)
    )

class WECTLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config: EctConfig):
        super().__init__()
        self.lin = (
            torch.linspace(-config.R, config.R, config.bump_steps)
            .view(-1, 1, 1)
        ).to(device=config.device)
        self.v = torch.vstack(
            [
                torch.sin(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                torch.cos(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
            ]
        ).to(config.device)
        
        self.device = config.device
        self.config = config


    def forward(self, data):
        out = torch.zeros(size=(self.config.bump_steps,data.batch.max()+1,self.config.num_thetas), device=self.device)
        ect = compute_wect(data, self.v, self.lin, out)
        return ect  

if __name__ == "__main__":
    batch_size=128
    device = 'cpu'
    # grid = torch.load("./precompute/grid.pt")
    # print(grid)
    # grid.edge_weights = torch.ones(1,2241,requires_grad=True)

    # Some custom data
    x = torch.tensor([[.2,.2],[.2,-.2],[-.2,.2]])
    edge_index = torch.tensor([[0,1,2],[1,2,0]])
    edge_weights = torch.tensor([1,1,1.0])
    data = Data(x=x,edge_index=edge_index,edge_weights=edge_weights)
    batch = Batch.from_data_list(batch_size*[data])
    layer = WECTLayer(EctConfig(num_thetas=32,bump_steps=32,device=device))
    res = layer(batch)

    import matplotlib.pyplot as plt
    plt.imshow(res[0].squeeze().detach().numpy())
    plt.show()

    