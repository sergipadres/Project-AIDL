from torch import nn

class MHSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads = 4):
        super().__init__()
        self.embed_dim = embed_dim

        found_heads = False
        for i in range(5,0,-1):
             if embed_dim%i == 0: 
                self.num_heads = i
                break

        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.qW = nn.Linear(self.embed_dim, self.embed_dim)
        self.kW = nn.Linear(self.embed_dim, self.embed_dim)
        self.vW = nn.Linear(self.embed_dim, self.embed_dim)
        self.vO = nn.Linear(self.embed_dim, self.embed_dim)

        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim *2),
            nn.GELU(),
            nn.Linear(self.embed_dim *2, self.embed_dim),
        )
        
    def forward(self, latent):

        x = latent
        x1 = self.norm1(latent)
        batch_size, n_tokens , _ = x.shape
        q = self.qW(x1).reshape(batch_size, self.num_heads, n_tokens, self.head_dim)
        k = self.kW(x1).reshape(batch_size, self.num_heads, n_tokens, self.head_dim)
        v = self.vW(x1).reshape(batch_size, self.num_heads, n_tokens, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
         
        attn = attn.softmax(dim=-1)

        x1 = (attn @ v).transpose(0, 1).reshape(batch_size, n_tokens, self.embed_dim)

        z = self.vO(x1) + x

        z1= self.norm2(z)

        out = z + self.ffn(z1)

        return out