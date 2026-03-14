import torch.nn as nn
import torchxrayvision as xrv

class SpatialVAE_XRV(nn.Module):
    def __init__(self, canales_in=512, latent_channels=4):
        super().__init__()
        experto = xrv.autoencoders.ResNetAE(weights="101-elastic")
        self.encoder = nn.Sequential(
            experto.conv1, experto.bn1, experto.relu, 
            experto.maxpool, experto.layer1, experto.layer2
        )
        self.mu_conv = nn.Conv2d(canales_in, latent_channels, 1)       
        self.logvar_conv = nn.Conv2d(canales_in, latent_channels, 1) 
        self.decoder_input = nn.Conv2d(latent_channels, canales_in, 1) 
        self.decoder = nn.Sequential(
            nn.Conv2d(canales_in, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, x):
        mu = self.mu_conv(self.encoder(x))
        return self.decoder(self.decoder_input(mu)), mu, None