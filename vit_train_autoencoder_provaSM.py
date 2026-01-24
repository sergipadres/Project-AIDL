# --- 1. CONFIGURACIÓ ---
!pip install medmnist

import torch
from torch import nn, optim
from medmnist import PneumoniaMNIST
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gc

# Neteja
torch.cuda.empty_cache()
gc.collect()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Usant dispositiu: {DEVICE}")

# --- 2. ENCODER (ViT)  ---

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
            x = self.proj(x)  # (Batch, embed_dim, H//patch_size, W//patch_size)
            x = x.flatten(2).transpose(1, 2)  # (Batch, num_patches, embed_dim)

            return x

class MHSelfAttentionBlock(nn.Module):
    def __init__(self, channels, embed_dim, num_heads=4, mlp_hidden=48, embed_hidden=48):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embed_hidden = embed_hidden
        self.head_dim = self.embed_hidden // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm((channels, self.embed_dim))
        
        self.preadapt = nn.Linear(self.embed_dim, self.embed_hidden)
        
        self.qW = nn.Linear(self.embed_hidden, self.embed_hidden)
        self.kW = nn.Linear(self.embed_hidden, self.embed_hidden)
        self.vW = nn.Linear(self.embed_hidden, self.embed_hidden)
        self.oW = nn.Linear(self.embed_hidden, self.embed_hidden)
        
        self.postadapt = nn.Linear(self.embed_hidden, self.embed_dim)
        
        self.norm2 = nn.LayerNorm((channels, self.embed_dim))
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden), 
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden), 
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden), 
            nn.SiLU(),
            nn.Linear(mlp_hidden, self.embed_dim), 
            nn.SiLU()
        )
        
    def forward(self, latent):
        latent = self.norm1(latent)
        latent = self.preadapt(latent)
        
        batch_size, _, _ = latent.shape
        
        q = self.qW(latent).reshape(batch_size, self.channels, self.num_heads, self.head_dim)
        k = self.kW(latent).reshape(batch_size, self.channels, self.num_heads, self.head_dim)
        v = self.vW(latent).reshape(batch_size, self.channels, self.num_heads, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        latent = (attn @ v).transpose(0, 1).reshape(batch_size, self.channels, self.embed_hidden)
        
        latent = self.postadapt(latent)
        latent = self.norm2(latent)

        x = latent + self.mlp(latent)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=64, num_heads=4, depth=6, mask_ratio=0.5, device = DEVICE):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mask_ratio = mask_ratio
        self.device = device
        
        #patchify
        self.patch_embed = PatchEmbedding(self.img_size,
                                          self.patch_size,
                                          self.in_channels,
                                          self.embed_dim)
        self.num_patches = self.patch_embed.num_patches 

        #positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        #self-attention blocks
        self.blocks = nn.ModuleList([MHSelfAttentionBlock(self.num_patches, self.embed_dim, self.num_heads) for _ in range(depth)])
     
        reduced_dims = torch.arange(self.embed_dim, 4, step=-20)
        
        self.mlp = nn.ModuleList([])
        for i in range(0, len(reduced_dims)-2):
            self.mlp.append(nn.Linear(reduced_dims[i], reduced_dims[i+1]))
            self.mlp.append(nn.SiLU())
        self.mlp.append(nn.Linear(reduced_dims[-2], 2))

    def forward(self, images):
        x = self.patch_embed(images) + self.pos_embed
        
        if self.training:
             B, N, D = x.shape
             size = int(N * 0.5)
             mask_indices = torch.randint(0, N, (B, size), device=x.device)
             batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
             x[batch_indices, mask_indices, :] = -1 ### ARA SI QUE ACTUALITZA X!

        #forward through self-attention blocks # Passem la X modificada
        for block in self.blocks: x = x + block(x)
            
        #ffn that reduces dimensionality    
        for block in self.mlp: x = block(x)

        
        batch_shape, _, _ = x.shape
        x = x.reshape(batch_shape, self.num_patches*2)
        return x

# --- 3. DECODER (CNN) - AQUEST ÉS EL NOU CANVI CLAU ---
# Aquest decoder usa convolucions per "pintar" millor els detalls

class CNNDecoder(nn.Module):
    def __init__(self, latent_input_dim, img_size=224):
        super().__init__()
        self.map_size = img_size // 16 # 14x14
        # Calculem quants canals necessitem per encabir el vector latent en un mapa de 14x14
        self.initial_channels = latent_input_dim // (self.map_size**2)
        
        # Si la divisió no és exacta, ajustem (seguretat)
        if self.initial_channels == 0: self.initial_channels = 1
        
        # Projectem el latent a la mida necessària si cal
        self.linear_proj = nn.Linear(latent_input_dim, self.initial_channels * self.map_size * self.map_size)

        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(self.initial_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, latent):
        B = latent.shape[0]
        x = self.linear_proj(latent)
        x = x.view(B, self.initial_channels, self.map_size, self.map_size)
        return self.decoder(x)


class ViTMaskedAutoencoderCNN(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=1, 
                 embed_dim=64, 
                 num_heads=4, 
                 encoder_depth=6, # canvi a 6 per velocitat
                 mask_ratio=0.5):
        super().__init__()
        
        # ENCODER (ViT)
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
            mask_ratio=mask_ratio
        )
        
        # Decoder CNN (Millora la visualització)
        # Calculem la mida del vector latent que surt de l'encoder
        num_patches = (img_size // patch_size) ** 2
        latent_dim = num_patches * 2 
        
        self.decoder = CNNDecoder(latent_input_dim=latent_dim, img_size=img_size)


    def encode(self, images):
        z = self.encoder(images)
        return z

    def decode(self, latent):
        images = self.decoder(latent)
        return images

    def forward(self, images):
        z = self.encode(images)   # Primer comprimim
        x_recon = self.decode(z)  # Després descomprimim
        return x_recon

# --- 4. ENTRENAMENT RAPID ---

img_size = 224
print("Carregant dades...")
data_train_raw = PneumoniaMNIST(split="train", download=True, size=img_size)
data_val_raw = PneumoniaMNIST(split="val", download=True, size=img_size)

## TOT EL DATASET(ocupa poc) A LA GPU!!!!!!!!!!
class TurboDataset(Dataset):
    def __init__(self, data):
        self.images = torch.tensor(data.imgs, dtype=torch.uint8).unsqueeze(1).float().div(255.0).to(DEVICE) #float() float32 (precisió estàndard).
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx]

train_dataset = TurboDataset(data_train_raw)
val_dataset = TurboDataset(data_val_raw)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ViTMaskedAutoencoderCNN(img_size=img_size).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')


epochs = 100  # <--- Farem que aprengui els detalls fins
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)

print("Iniciant entrenament i amb millor qualitat visual...")

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    batch_losses = []
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    
    for imgs in loop:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'): 
            reconstructed = model(imgs)
            loss = loss_fn(reconstructed, imgs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        batch_losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
            
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for imgs in val_loader:
            with autocast():
                recon = model(imgs)
                val_batch_losses.append(loss_fn(recon, imgs).item())
    
    val_loss = np.mean(val_batch_losses)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}")

# --- 5. RESULTATS ---
print("\nVisualitzant resultats...")
model.eval()
with torch.no_grad():
    sample_imgs = val_dataset.images[:5] 
    with torch.amp.autocast('cuda'):
        reconstructed = model(sample_imgs)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(sample_imgs[i].cpu().float().squeeze(), cmap="gray")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].cpu().float().squeeze(), cmap="gray")
    axes[1, i].set_title("Reconstruït (Híbrid)")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()

# Guardem l'Encoder per utilitzar-lo a la següent fase
torch.save(model.encoder.state_dict(), "encoder_vit_ready.pth")
print(" Encoder guardat!")