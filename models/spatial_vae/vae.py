import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SpatialVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder ResNet18
        self.encoder = smp.Unet("resnet18", in_channels=1, encoder_weights="imagenet").encoder
        
        # Latent 4x14x14
        self.mu_conv = nn.Conv2d(256, 4, 1)
        self.logvar_conv = nn.Conv2d(256, 4, 1)
        
        # Decoder
        self.decoder_input = nn.Conv2d(4, 256, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid() 
        )

    def encode(self, x):
        features = self.encoder(x)
        x_enc = features[-2]
        mu, log_var =  self.mu_conv(x_enc), self.logvar_conv(x_enc)
        log_var = torch.clamp(log_var, -10, 10)
        return mu, log_var

    def decode(self, mu, log_var):
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        out = self.decoder_input(z)
        out = self.decoder(out)
        return out

    def sample(self, mu, log_var):
        z = torch.normal(mu, torch.exp(0.5 * log_var))
        out = self.decoder_input(z)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mu, log_var = self.encode(x)
        recons = self.decode(mu, log_var)
        return recons, mu, log_var


