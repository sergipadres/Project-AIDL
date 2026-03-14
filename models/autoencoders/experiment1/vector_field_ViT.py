
import torch
from torch import nn
from mhsa import MHSelfAttentionBlock
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimestepEmbedding(nn.Module):
    def __init__(self, timestep_embed_dim = 96, hidden_dim = 128):
        super().__init__()

        self.timestep_embed_dim = timestep_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, timestep_embed_dim)
        )

    def forward(self, timestep):

        emb = -torch.log(torch.tensor(0.5))*timestep
        emb = torch.tensor([torch.sin(emb), torch.cos(emb)]).to(device)
        out = self.mlp(emb)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=1,
                 embed_dim=32):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (Batch, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (Batch, embed_dim, num_patches)
        x = x.permute(0,2,1)  # (Batch, num_patches, embed_dim)
        return x

def embed3d(f_maps: torch.Tensor):

    orig_shape = f_maps.shape
    _, n_channels, n_patches = orig_shape
    n_patches_per_axis = int(n_patches**0.5)  
    f_maps = f_maps.reshape(-1,n_channels, n_patches_per_axis,n_patches_per_axis)
    pairs_dim = n_channels // 2
    theta = torch.tensor([10000.0 ** (-2 * i / n_channels) for i in range(pairs_dim)])


    for i in range(pairs_dim):

        c1, c2 = 2*i, 2*i + 1
        phi = torch.arange(n_patches_per_axis) * theta[i]
        cos_phi = torch.cos(phi).to(device)
        sin_phi = torch.sin(phi).to(device)
        for row in range(n_patches_per_axis):

            x_row = f_maps[:, c1, row, :]
            y_row = f_maps[:, c2, row, :]
            rotated_x = x_row * cos_phi - y_row * sin_phi
            rotated_y = x_row * sin_phi + y_row * cos_phi

            f_maps[:, c1, row, :] = rotated_x
            f_maps[:, c2, row, :] = rotated_y

        for col in range(n_patches_per_axis):

            x_col = f_maps[:, c1, :, col]
            y_col = f_maps[:, c2, :, col]
            rotated_x = x_col * cos_phi - y_col * sin_phi
            rotated_y = x_col * sin_phi + y_col * cos_phi

            f_maps[:, c1, :, col] = rotated_x
            f_maps[:, c2, :, col] = rotated_y 
     
    f_maps = f_maps.flatten(-2)
    return f_maps

class Unpatchify(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dim,
                 target_img_size,
                 target_channels):
        super().__init__()

        self.patch_h = self.patch_w = int(num_patches**0.5)
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 48, 4, 2, 1),
            nn.LayerNorm((28,28)),
            nn.SiLU(),
            nn.ConvTranspose2d(48,24,4,2,1),
            nn.LayerNorm((56,56)),
            nn.SiLU(),
            nn.ConvTranspose2d(24,12,4,2,1),
            nn.LayerNorm((112,112)),
            nn.SiLU(),
            nn.ConvTranspose2d(12,target_channels,4,2,1),
            nn.SiLU())


    def forward(self, patches):

        patches = patches.permute(0,2,1)
        patches = patches.reshape(-1, self.embed_dim, self.patch_h, self.patch_w)
        patches = self.conv(patches)
        patches = torch.clamp(patches,0,1)
        return patches


class TimestepVisionTransformer(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 in_channels = 1,
                 attention_depth = 2,
                 mlp_depth = 3,
                 embed_dim = 96,
                 num_heads = 4,
                 target_channels =  1
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.attention_depth = attention_depth
        self.mlp_depth = mlp_depth
        self.embed_dim = embed_dim

        self.timestep_embed = TimestepEmbedding(embed_dim)

        #patchify:
        #Generates (img_size/patch_size)**2 patches of dimension embed_dim
        self.patch_embed = PatchEmbedding(img_size,
                                        patch_size,
                                        in_channels,
                                        embed_dim)

        self.num_patches = self.patch_embed.num_patches


        n = self.num_patches

        self.attn = nn.Sequential()
        for i in range(self.attention_depth):
            self.attn.append(MHSelfAttentionBlock(embed_dim = embed_dim))

        self.final_norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential()
        for _ in range(self.mlp_depth):
            self.mlp.append(nn.Linear(embed_dim, embed_dim))
            self.mlp.append(nn.SiLU())

        self.unpatchify = Unpatchify(self.num_patches,
                                     embed_dim,
                                     img_size,
                                     in_channels)

    def forward(self, x, timestep):

        time = self.timestep_embed(timestep)

        patches = self.patch_embed(x)

        patches = patches.permute(0,2,1)
        patches = embed3d(patches)
        patches = patches.permute(0,2,1)

        for block in self.attn:
            patches = block(patches)

        patches = patches + time

        for block in self.mlp:
            patches = block(patches)

        out = self.unpatchify(patches)

        return out