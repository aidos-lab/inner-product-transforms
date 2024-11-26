import torch 
from torch import nn
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt


from layers.directions import generate_directions
from layers.ect import EctLayer, EctConfig



def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)





class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(8,16,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(16,32,kernel_size=3,stride=3),
        )

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*2*32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        return x


if __name__ == "__main__":
    v = generate_directions(num_thetas=64,d=2,device="cuda:0")
    layer = EctLayer(EctConfig(num_thetas=64,bump_steps=64),v=v)
    batch = Batch.from_data_list([Data(x=torch.rand(size=(5,2))),Data(x=torch.rand(size=(5,2)))]).cuda()
    ect = layer(batch,batch.batch).movedim(-1,-2)
    model = Model().cuda()
    out = model(ect.unsqueeze(1)).cpu().detach().squeeze().numpy()
    print(out.shape)

    # plt.imshow(ect[0].detach().cpu().numpy())
    # plt.show()
    # plt.scatter(out[:,0],out[:,1])
    # plt.show()


