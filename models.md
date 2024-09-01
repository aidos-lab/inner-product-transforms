
Works very well. 
better results with 256 as latent 
and 3 layers, but currents set up is very good 
conv1 layer can not be increased, for some reason.

the currently implemented max works very well, 
much better then the sum 

When implementing, make sure you do the 
ect right otherwise it does not learn.

also, axis of max can not be exchanged. 


```{python}
class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # self.layer = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64**2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 256),
        # )
        
        self.conv = nn.Sequential(
            nn.Conv1d(1,512,kernel_size=64,stride=64),
            nn.ReLU(),
        )

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1).unsqueeze(1)
        x = self.conv(x).max(axis=-2)[0]
        x = self.layer(x)
        return x

```
