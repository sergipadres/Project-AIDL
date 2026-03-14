#v8 Spatial VAE pur (sense skip connections) amb vgg_loss i resolució pujada a 14x14  data: 14_02_2026  canvis Sandra Márquez

# Forcem l'actualitzacio del pip i instal·lem especificant els noms exactes
get_ipython().getoutput("pip install --upgrade pip")
get_ipython().getoutput("pip install segmentation-models-pytorch medmnist")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from medmnist import PneumoniaMNIST
import os
import gc

# --- 1. CONFIGURACIÓ ---
CONFIG = {
    "IMG_SIZE": 224,
    "BATCH_SIZE": 16,       # Segur per a 14GB de GPU
    "ACCUM_STEPS": 2,       # Acumulació per simular Batch=32
    "LATENT_CHANNELS": 4,   # Latent 4x14x14
    "EPOCHS": 20,           # Suficient per veure resultats
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

# --- 4. MODEL AMB SIGMOID  ---
class SpatialVAE_Final(nn.Module):
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
            nn.Sigmoid()  # <--- LA PEÇA CLAU: Força sortida 0-1
        )

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, x):
        features = self.encoder(x)
        x_enc = features[-2] # 14x14
        mu, log_var = self.mu_conv(x_enc), self.logvar_conv(x_enc)
        log_var = torch.clamp(log_var, -10, 10)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(self.decoder_input(z))
        return recon, mu, log_var

# --- 5. VISUALITZACIÓ AUTOMÀTICA (SANITY CHECK+ MORPH) ---
#def check_results(model, loader):
 #   model.eval()
  #  imgs, labels = next(iter(loader))
   # imgs = imgs.to(CONFIG["DEVICE"])
    
    # Buscar Sa i Malalt
    #try:
      #  idx_H = (labels == 0).nonzero(as_tuple=True)[0][0]
      #  idx_P = (labels == 1).nonzero(as_tuple=True)[0][0]
    #except: idx_H, idx_P = 0, 1
    
    #with torch.no_grad():
     #   _, mu_H, _ = model(imgs[idx_H].unsqueeze(0))
     #   _, mu_P, _ = model(imgs[idx_P].unsqueeze(0))
        
        # --- TEST 1: SANITY CHECK ---
      #  latent_frank = mu_H.clone()
       # latent_frank[:,:,:,7:] = mu_P[:,:,:,7:] # Meitat dreta malalta
        
        #recon_H = model.decoder(model.decoder_input(mu_H))
        #recon_P = model.decoder(model.decoder_input(mu_P))
        #recon_F = model.decoder(model.decoder_input(latent_frank))
        
        # Plot
        #fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #ax[0].imshow(recon_H.cpu().squeeze(), cmap='gray', vmin=0, vmax=1); ax[0].set_title("Sa")
        #ax[1].imshow(recon_P.cpu().squeeze(), cmap='gray', vmin=0, vmax=1); ax[1].set_title("Pneumònia")
        #ax[2].imshow(recon_F.cpu().squeeze(), cmap='gray', vmin=0, vmax=1); ax[2].set_title("Sanity check")
        #plt.suptitle("Test de Transplantament (Si es veu bé, tenim èxit)")
        #plt.savefig(f"{CONFIG['SAVE_DIR']}/sanity_check_final.png")
        #plt.show()

# --- 6. BUCLE D'ENTRENAMENT ---
if __name__ == "__main__":
    gc.collect(); torch.cuda.empty_cache()
    loaders = get_dataloaders()
    model = SpatialVAE_Final().to(CONFIG["DEVICE"])
    vgg_loss = VGGPerceptualLoss().to(CONFIG["DEVICE"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    scaler = torch.cuda.amp.GradScaler() # Si tens pytorch vell
    # scaler = torch.amp.GradScaler('cuda') # Si tens pytorch nou
    
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
            
        print(f"Ep {epoch+1}: Avg Loss = {total_loss/len(loaders['train']):.4f}")
        
        # Validació visual ràpida a la meitat i al final
        #if (epoch+1) % 5 == 0:
         #   check_results(model, loaders['val'])
            
    # GUARDAR
    torch.save(model.state_dict(), f"{CONFIG['SAVE_DIR']}/vae_final_sigmoid.pth")
    
    # EXTRAURE LATENTS
    print(" Guardant latents...")
    model.eval()
    for split in ['train', 'val']:
        latents, labels = [], []
        for x, y in tqdm(loaders[split]):
            with torch.no_grad():
                _, mu, _ = model(x.to(CONFIG["DEVICE"]))
                latents.append(mu.cpu().numpy())
                labels.append(y.cpu().numpy())
        np.save(f"{CONFIG['SAVE_DIR']}/latents_{split}.npy", np.concatenate(latents))
        np.save(f"{CONFIG['SAVE_DIR']}/labels_{split}.npy", np.concatenate(labels))
        
    print("FI VAE.")



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from medmnist import PneumoniaMNIST
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

# --- 1. CONFIGURACIÓ  ---
CONFIG = {
    "IMG_SIZE": 224,
    "BATCH_SIZE": 16,  # Aquí podem pujar-ho una mica per fer inferència
    "LATENT_CHANNELS": 4,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SAVE_DIR": "./artifacts_final",
    "MODEL_PATH": "./artifacts_final/vae_final_sigmoid.pth"
}

# --- 2. DEFINICIÓ DEL MODEL (Necessària per carregar els pesos) ---
class SpatialVAE_HighRes(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        aux_model = smp.Unet("resnet18", in_channels=1, encoder_weights=None) # No cal baixar pesos d'imagenet ara
        self.encoder = aux_model.encoder
        self.mu_conv = nn.Conv2d(256, latent_channels, kernel_size=1)
        self.logvar_conv = nn.Conv2d(256, latent_channels, kernel_size=1)
        self.decoder_input = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def reparameterize(self, mu, log_var):
        return mu # Per fer inferencia nomes fem servir la mitjana

    def forward(self, x):
        x_encoded = self.encoder(x)[-2]
        mu = self.mu_conv(x_encoded)
        recon = self.decoder(self.decoder_input(mu))
        return recon, mu, None

# --- 3. CARREGAR DADES ---
class TurboDataset(Dataset):
    def __init__(self, data):
        self.images = torch.tensor(data.imgs, dtype=torch.uint8).unsqueeze(1).float().div(255.0).to(CONFIG["DEVICE"])
        self.labels = torch.tensor(data.labels, dtype=torch.long).to(CONFIG["DEVICE"])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

def get_val_loader():
    print(" Carregant dades de validació...")
    data_val = PneumoniaMNIST(split="val", download=True, size=CONFIG["IMG_SIZE"])
    return DataLoader(TurboDataset(data_val), batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

# --- 4. GENERADOR DE GRÀFICS ---
def recover_plots():
    # A. Instanciar i Carregar
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print(f" ERROR: No trobo l'arxiu {CONFIG['MODEL_PATH']}")
        return

    print(f"Carregant model des de {CONFIG['MODEL_PATH']}...")
    model = SpatialVAE_HighRes(latent_channels=CONFIG["LATENT_CHANNELS"]).to(CONFIG["DEVICE"])
    state_dict = torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"])
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    loader = get_val_loader()

    # B. Plot de Reconstrucció (Igual que abans)
    print(" Generant comparativa visual...")
    imgs, labels = next(iter(loader))
    imgs = imgs.to(CONFIG["DEVICE"])

    try:
        idx_H = (labels == 0).nonzero(as_tuple=True)[0][0]
        idx_P = (labels == 1).nonzero(as_tuple=True)[0][0]
    except:
        idx_H, idx_P = 0, 1

    with torch.no_grad():
        rec_H, _, _ = model(imgs[idx_H].unsqueeze(0))
        rec_P, _, _ = model(imgs[idx_P].unsqueeze(0))

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    
    def show(ax, img, title, diff=False):
        im = img.cpu().squeeze().numpy()
        cmap = 'inferno' if diff else 'gray'
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    show(ax[0,0], imgs[idx_H], "SA (Original)")
    show(ax[0,1], rec_H, "SA (Reconstrucció)")
    show(ax[0,2], torch.abs(imgs[idx_H]-rec_H), "Error", diff=True)
    show(ax[1,0], imgs[idx_P], "PNEUMÒNIA (Original)")
    show(ax[1,1], rec_P, "PNEUMÒNIA (Reconstrucció)")
    show(ax[1,2], torch.abs(imgs[idx_P]-rec_P), "Error", diff=True)

    plt.tight_layout()
    plt.show()

    # --- C. Plot PCA AMB DISTÀNCIA ---
    print(" Calculant PCA i Distàncies...")
    latents, labels_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i > 50: break 
            _, mu, _ = model(x)
            latents.append(mu.cpu().view(mu.size(0), -1).numpy())
            labels_list.append(y.cpu().numpy())
    
    X = np.concatenate(latents)
    y = np.concatenate(labels_list).squeeze() 

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # --- NOVES LÍNIES DE CÀLCUL ---
    # 1. Calculem els centres
    center_H = X_pca[y==0].mean(axis=0)
    center_P = X_pca[y==1].mean(axis=0)
    
    # 2. Calculem la distància
    dist = np.linalg.norm(center_H - center_P)
    
    # 3. Pintem el gràfic
    plt.figure(figsize=(10, 8))
    
    # Punts normals
    plt.scatter(X_pca[y==0,0], X_pca[y==0,1], c='dodgerblue', alpha=0.4, label='Normal', s=20)
    # Punts pneumònia
    plt.scatter(X_pca[y==1,0], X_pca[y==1,1], c='crimson', alpha=0.4, label='Pneumonia', s=20)
    
    # CENTROIDES (Les X grans)
    plt.scatter(*center_H, c='navy', s=200, marker='X', edgecolors='white', linewidth=2, label='Centre Normal')
    plt.scatter(*center_P, c='darkred', s=200, marker='X', edgecolors='white', linewidth=2, label='Centre Pneumonia')
    
    # Línia que uneix els centres
    plt.plot([center_H[0], center_P[0]], [center_H[1], center_P[1]], 'k--', alpha=0.5)

    plt.title(f"PCA High-Res (14x14) | Distància entre Centres: {dist:.4f}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Distància PCA calculada: {dist:.4f}")

if __name__ == "__main__":
    recover_plots()


# COMPROVACIO DE DISTANCIES VEI MES PROPER I DUPLICATS


import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def check_final_latents(file_path):
    try:
        # 1. Carregar dades
        data = np.load(file_path)
        
        # Aplanar per calcular distàncies (N, 4, 14, 14) -> (N, 784)
        data_flat = data.reshape(data.shape[0], -1)
        
        # 2. Cerca de Duplicats
        unique_data = np.unique(data_flat, axis=0)
        duplicates = len(data_flat) - len(unique_data)
        
        print(f"   Shape original: {data.shape}")
        print(f"   Duplicats exactes: {duplicates} (de {len(data)})")
        if duplicates > 0:
            print("   ( Els duplicats probablement venen d'imatges idèntiques al dataset original)")
        
        # 3. Distància al Veí Més Proper (Línia base)
        print(" Calculant distàncies ...")
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data_flat)
        distances, _ = nbrs.kneighbors(data_flat)
        
        # La columna 0 és distància a si mateix (0), la 1 és al veí
        nn_dists = distances[:, 1]
        
        mitjana = np.mean(nn_dists)
        minima = np.min(nn_dists[nn_dists > 0])
        
        print(f"\n RESULTATS DE DISTÀNCIA:")
        print(f"   Distància Mitjana entre Veïns: {mitjana:.4f} ")
        print(f"   Distància Mínima (no zero): {minima:.4f}")
        
        # 4. Mostrar Histograma
        plt.figure(figsize=(8, 4))
        plt.hist(nn_dists, bins=50, color='mediumseagreen', edgecolor='black')
        plt.axvline(mitjana, color='red', linestyle='dashed', linewidth=2, label=f'Mitjana: {mitjana:.2f}')
        plt.title("Distribució de Distàncies a l'Espai Latent Final (14x14)")
        plt.xlabel("Distància Euclidiana al Veí més proper")
        plt.ylabel("Freqüència (Pacients)")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f" Error en llegir o processar l'arxiu: {e}")

if __name__ == "__main__":
    # Posar la ruta on ha guardat els latents
    check_final_latents("artifacts_final/latents_train.npy")
