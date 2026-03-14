import torch
from torch import nn

class Gamma(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.timestep_embed = nn.Linear(1,2*latent_dim)
        self.mlp = nn.ModuleList([
                                nn.Linear(2*latent_dim, 2*latent_dim),
                                nn.Tanh(),
                                nn.Linear(2*latent_dim, 2*latent_dim),
                                nn.Tanh(),
                                ])
        self.output = nn.Sequential(nn.Linear(2*latent_dim, latent_dim),
                                    nn.Tanh(),
                                    nn.Linear(latent_dim, latent_dim),
                                    )


    def forward(self, x0_latent, x1_latent, timestep):
        t = timestep.unsqueeze(0)
        t = self.timestep_embed(t)
        x = torch.cat((x0_latent, x1_latent), dim=1)
        for i, block in enumerate(self.mlp):
            x = block(x)
        x = x + t
        x = self.output(x)
        return x


