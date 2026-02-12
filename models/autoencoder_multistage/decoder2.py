import torch
from torch import nn
from mhsa import MHSelfAttentionBlock
from torch.nn.functional import interpolate, silu, relu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels//16),
            nn.ReLU(),
            nn.Linear(channels//16, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        attention = self.squeeze(x).flatten(-3)
        attention = self.excitation(attention).view(batch, channels, 1, 1)
        x = x * attention
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_avg = torch.cat([max_out, avg_out], dim = 1)
        attention = self.conv(max_avg)
        attention = self.sigmoid(attention)
        x = x * attention
        return x

class CBAM(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channel_att = SqueezeExcitation(channels)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(ca)
        return sa


class ResUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, target_height):
        super().__init__()
        
        self.target_height = target_height
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attention = CBAM(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = interpolate(x, size=self.target_height, mode="bilinear")
        residual = self.shortcut_conv(residual)

        x = interpolate(x, size=self.target_height, mode="bilinear")
        x = self.conv1(x)
        x = self.bn1(x)
        x = silu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = silu(x)

        out = x + residual

        out = self.attention(out)
        out = self.relu(out)

        return out




class Decoder(nn.Module):
            
    def __init__(self,
                 latent_dim,
                 num_generated_fmaps = 512,
                 fmap_height = 7,
                 target_channels = 1,
                 target_height = 224,
                 upscale_depth = 12,
                 ):

        super().__init__()
        self.num_generated_fmaps = num_generated_fmaps
        self.fmap_height = fmap_height
        #FIRST STAGE OF THE DECODER
        #At this stage we go from the latent produced by the encoder,
        #To a sequence of feature maps (channels) in an autoregressive way
        #Layers for autoregressive generation of feature maps
        
        self.make_feature_maps = nn.Linear(latent_dim, num_generated_fmaps*(fmap_height**2))

        # #SECOND STAGE OF THE DECODER
        # #Here we have feature maps of 14x14
        # #We want to upscale them until we form an image


        # #Calculate channel reduction schedule given depth
        # #this gives each channel size for every block
        # upscale_depth = len(target_height_schedule)
        channel_schedule = torch.linspace(num_generated_fmaps, target_channels, upscale_depth+1, dtype=torch.int16)
        in_channel_schedule = channel_schedule[:-1]
        out_channel_schedule = channel_schedule[1:]

        # #Calculate height upscaling schedule given depth
        # #We want to upscale from original height to target_height
        target_height_schedule = torch.linspace(fmap_height, target_height, upscale_depth ,dtype=torch.int16)

        self.blocks_upscale = nn.Sequential()
        for i in range(upscale_depth):
            self.blocks_upscale.append(ResUpBlock(in_channels = in_channel_schedule[i],
                                                  out_channels = out_channel_schedule[i],
                                                  target_height = target_height_schedule[i]
                                                 ) 
                                       )



    def forward(self, latent):

        #first we generate feature maps
        f_maps = self.make_feature_maps(latent)
        f_maps = f_maps.reshape(-1, self.num_generated_fmaps, self.fmap_height, self.fmap_height)

        #then, we scale them
        for block in self.blocks_upscale:
            f_maps = block(f_maps)

        return f_maps