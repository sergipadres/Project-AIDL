import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader, Dataset
from vae import SpatialVAE
import torchvision.models as models
import segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm import tqdm
import os
import gc
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓ ---
CONFIG = {
    "IMG_SIZE": 224,
    "BATCH_SIZE": 20,       # Segur per a 14GB de GPU
    "ACCUM_STEPS": 2,       # Acumulació per simular Batch=32
    "LATENT_CHANNELS": 4,   # Latent 4x14x14
    "EPOCHS": 100,           # Suficient per veure resultats
    "LR": 1e-4,
    "VGG_WEIGHT": 0.05,     # Ajustat per evitar que domini massa
    "KL_WEIGHT": 0.00025,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SAVE_DIR": "./artifacts_final"
}
os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

# --- 2. DATASET ---
class TurboDataset(Dataset):
    def __init__(self, data):
        # Normalitzem a 0-1
        self.images = torch.tensor(data.imgs, dtype=torch.uint8).unsqueeze(1).float().div(255.0).to(CONFIG["DEVICE"])
        self.labels = torch.tensor(data.labels, dtype=torch.long).to(CONFIG["DEVICE"])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

def get_dataloaders():
    print("Carregant dades...")
    data_train = PneumoniaMNIST(split="train", download=True, size=CONFIG["IMG_SIZE"])
    data_val = PneumoniaMNIST(split="val", download=True, size=CONFIG["IMG_SIZE"])
    return {
        'train': DataLoader(TurboDataset(data_train), batch_size=CONFIG["BATCH_SIZE"], shuffle=True),
        'val':   DataLoader(TurboDataset(data_val), batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    }

# --- 3. VGG LOSS Amb Normalització ImageNet ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        for param in self.parameters(): param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # 1. Convertir BW -> RGB
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # 2. Normalitzar estil ImageNet + NITID
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0
        x, y = input, target
        for block in self.blocks:
            x, y = block(x), block(y)
            loss += nn.L1Loss()(x, y)
        return loss

if __name__ == "__main__":
    gc.collect(); torch.cuda.empty_cache()
    loaders = get_dataloaders()
    model = SpatialVAE().to(CONFIG["DEVICE"])
    vgg_loss = VGGPerceptualLoss().to(CONFIG["DEVICE"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    #scaler = torch.cuda.amp.GradScaler() # Si tens pytorch vell
    scaler = torch.amp.GradScaler('cuda') # Si tens pytorch nou
    
    print(" ENTRENANT TOT PLEGAT (Amb Sigmoid)...")
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        loop = tqdm(loaders['train'], leave=False)
        total_loss = 0
        
        for i, (imgs, _) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                recon, mu, log_var = model(imgs)
                # Usem MSE o L1. Com que tenim Sigmoid (0-1), L1 va perfecte.
                loss_pixel = nn.L1Loss()(recon, imgs)
                loss_vgg = vgg_loss(recon, imgs) * CONFIG["VGG_WEIGHT"]
                loss_kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                
                loss = loss_pixel + loss_vgg + (CONFIG["KL_WEIGHT"] * loss_kl)
                loss = loss / CONFIG["ACCUM_STEPS"]
            
            scaler.scale(loss).backward()
            
            if (i+1) % CONFIG["ACCUM_STEPS"] == 0:
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item() * CONFIG["ACCUM_STEPS"]
            loop.set_postfix(Loss=f"{loss.item()*CONFIG['ACCUM_STEPS']:.4f}")
            
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss/len(loaders['train']):.4f}")
            
        #Validation
        #This does manual inspection of the reconstructed images
        model.eval()
        val_losses = []
        fig, axs = plt.subplots(2, 8, figsize=(40,10))

        for i, image_idx in enumerate(torch.randint(low=0,  high = len(loaders["val"].dataset) - 1, size=(8,))):

            original_img, _ = loaders["val"].dataset[image_idx]

            axs[0,i].imshow(original_img.cpu().squeeze().numpy(), cmap = "grey")
            axs[0,i].set_title("original")

            
            with torch.no_grad():

                mu, log_var = model.encode(original_img
                                                .unsqueeze(0))

                y_hat_val = model.sample(mu,log_var).squeeze()

                val_loss = nn.L1Loss()(y_hat_val, original_img)
                val_losses.append(val_loss.cpu().detach().numpy())
                axs[1,i].imshow(y_hat_val.cpu().detach().numpy(), cmap="grey")
                axs[1,i].set_title("reconstruction")
        
        plt.savefig(f'./reconstructions/epoch-{epoch:03}.png')
        plt.close()
        print(f"\n L1 Validation Loss at epoch {epoch}: {np.mean(val_losses)}")
    
        # GUARDAR
        if (epoch % 5 == 0):
            torch.save(model.state_dict(), f"{CONFIG['SAVE_DIR']}/vae_final_sigmoid-{epoch+1}.pth")