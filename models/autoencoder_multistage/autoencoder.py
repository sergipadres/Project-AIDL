from torch import nn
from encoder import ViTEncoder
from decoder2 import Decoder

class ViTMaskedAutoencoder(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 in_channels = 1,
                 embed_dim = 128,
                 latent_dim = 49*2,
                 num_heads = 4,
                 encoder_attention_depth=3,
                 encoder_mlp_depth=4,
                 mask_ratio=0.5,
                 ):
        super().__init__()
        self.encoder = ViTEncoder(img_size = img_size,
                                  patch_size= patch_size,
                                  in_channels = in_channels,
                                  embed_dim = embed_dim,
                                  num_heads = num_heads,
                                  mask_ratio = mask_ratio,
                                  latent_dim=latent_dim,
                                  attention_depth = encoder_attention_depth,
                                  mlp_depth = encoder_mlp_depth
                                  )
        self.decoder = Decoder(latent_dim=latent_dim)


    def encode(self, images):
        z = self.encoder.forward(images)
        return z

    def decode(self, latent):
        preds = self.decoder.forward(latent)
        return preds
