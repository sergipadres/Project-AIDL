import torch
from torch import nn
from torch.nn.functional import silu

from mhsa import MHSelfAttentionBlock
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=8,
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


#embed patches function
#we embed for every channel pair its rows and columns
#this is a simplification of rope, 
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

class ViTEncoder(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 14,
                 in_channels = 1,
                 attention_depth = 10,
                 mlp_depth = 6,
                 embed_dim = 128,
                 num_heads = 4,
                 mask_ratio=0.50,
                 latent_dim = 196,  #will be 8,4,4
                 device = device
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.attention_depth = attention_depth
        self.mlp_depth = mlp_depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.latent_dim = latent_dim
        self.device = device

        #patchify:
        #Generates (img_size/patch_size)**2 patches of dimension embed_dim
        self.patch_embed = PatchEmbedding(self.img_size,
                                        self.patch_size,
                                        self.in_channels,
                                        self.embed_dim)

        self.num_patches = self.patch_embed.num_patches

        self.neg_mask = nn.Parameter(torch.zeros(1, self.embed_dim, dtype=torch.float32))

        n = self.num_patches
        e = self.embed_dim
        e_dim = [e,n] * (self.attention_depth)
        e_dim = torch.tensor(e_dim).flatten(0).to(device)
        e_dim = e_dim[:self.attention_depth].reshape(self.attention_depth,1)

        self.attn = nn.Sequential()
        for i in range(self.attention_depth):
            self.attn.append(MHSelfAttentionBlock(embed_dim = e_dim[i]))

        self.final_norm = nn.LayerNorm(e_dim[-1])

        self.mlp = nn.Sequential()
        for _ in range(self.mlp_depth):
            self.mlp.append(nn.Linear(e_dim[-1], e_dim[-1]))
            self.mlp.append(nn.SiLU())

        self.final_reduction = nn.Sequential(nn.Linear(self.embed_dim*self.num_patches, self.latent_dim),
                                            nn.SiLU())


    def forward(self, images):

        patches = self.patch_embed(images) #(batch, num_patches, embed_dim)
 
        #randomly mask images
        if self.training:
            num_patches = self.num_patches
            size = int(num_patches*self.mask_ratio)
            batch_size = patches.shape[0]
            masked_indexes = torch.stack([torch.randint(low=0, 
                                                        high=num_patches,
                                                        size=(size,))
                                        for  _ in range(batch_size)]).flatten().to(device)
                  
            batch_indexes = torch.arange(batch_size).repeat(1,size).flatten().to(device)
            
            patches[batch_indexes, masked_indexes,:] = self.neg_mask

        
        x = patches.permute(0,2,1)
        x = embed3d(x)
        x = x.permute(0,2,1)
        #forward through self-attention blocks
        for block in self.attn:
            x = block(x)
            x = x.permute(0,2,1)

        x = x.permute(0,2,1)
        x = self.final_norm(x)

        for block in self.mlp:
            x = block(x)

        x = x.flatten(1)
        x = self.final_reduction(x)
        
        return x
