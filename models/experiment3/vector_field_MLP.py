import torch
from torch import nn

class TimestepEmbedding(nn.Module):
    def __init__(self, timestep_embed_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, timestep_embed_dim)
        )

    def forward(self, timestep):
        device = timestep.device
        #emb = -torch.log(torch.tensor(0.5))*timestep
        emb = timestep
        emb = torch.tensor([torch.sin(emb*360), torch.cos(emb*360)]).to(device)
        out = self.mlp(emb)
        return out



class vector_field_MLP(nn.Module):

    def __init__(self, latent_dim=98, time_embed_dim=32):
        super().__init__()

        self.timestep_embedding = TimestepEmbedding(time_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim+time_embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim),
            nn.GELU()            
        )

    def forward(self, x, timestep):
        t_embed = self.timestep_embedding(timestep).expand(x.shape[0],-1)
        embed = torch.cat((x,t_embed), dim=1)
        out = self.net(embed)
        return out

# import torch
# from torch import nn
# from torch.nn.functional import silu

# class vector_field_MLP(nn.Module):
#     def __init__(self, latent_dim=64, hidden_dim = 128, depth=6):
#         super().__init__()
        
#         self.timestep_embed = nn.Sequential(nn.Linear(1,hidden_dim),
#                                             nn.SiLU(),
#                                             nn.Linear(hidden_dim,1))


#         self.input_layer = nn.Linear(latent_dim*2 + 1, hidden_dim)
        
#         self.mlp = nn.Sequential()
#         for _ in range (depth):
#             self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
#             self.mlp.append(nn.SiLU())
        
#         self.output_layer = nn.Linera(hidden_dim, latent_dim)


#     def forward(self, interpolant, timestep):

#         batch, _ = x0.size
#         time_embed = self.timestep_embed(timestep).unsqueeze(0).expand(batch, -1)

#         interp_time = torch.cat([interpolant, timestep], dim = 1)

#         out = self.input_layer(interp_time)
#         out = self.mlp(out)
#         out = self.output_layer(out)
#         out = silu(out)

#         return out