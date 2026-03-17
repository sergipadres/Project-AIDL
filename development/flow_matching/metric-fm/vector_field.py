import torch
from torch import nn
from torch.nn.functional import silu

class VectorField(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim = 128, depth=6):
        super().__init__()
        
        self.timestep_embed = nn.Sequential(nn.Linear(1,hidden_dim),
                                            nn.SiLU(),
                                            nn.Linear(hidden_dim,1))


        self.input_layer = nn.Linear(latent_dim*2 + 1, hidden_dim)
        
        self.mlp = nn.Sequential()
        for _ in range (depth):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU())
        
        self.output_layer = nn.Linera(hidden_dim, latent_dim)


    def forward(self, interpolant, timestep):

        batch, _ = x0.size
        time_embed = self.timestep_embed(timestep).unsqueeze(0).expand(batch, -1)

        interp_time = torch.cat([interpolant, timestep], dim = 1)

        out = self.input_layer(interp_time)
        out = self.mlp(out)
        out = self.output_layer(out)
        out = silu(out)

        return out