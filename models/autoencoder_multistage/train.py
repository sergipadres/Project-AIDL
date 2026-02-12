import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from autoencoder import ViTMaskedAutoencoder
from medmnist import PneumoniaMNIST
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lpips
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data
img_size=224

data_pneumonia = PneumoniaMNIST(split="train", download=True, size=img_size)
data_pneumonia_val = PneumoniaMNIST(split="val", download=True, size=img_size)
images = torch.tensor(data_pneumonia.imgs, dtype=torch.uint8).to(torch.float)/255
images_val = torch.tensor(data_pneumonia_val.imgs, dtype=torch.uint8).to(torch.float)/255

class CustomImageDataset(Dataset):
    def __init__(self, images):
        #add channel dimension
        self.images = images.unsqueeze(1)
        self.images = self.images.to(device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx,:,:,:]

training_images = CustomImageDataset(images)
validation_images = CustomImageDataset(images_val)
batch_size = 16
train_dataloader = DataLoader(training_images, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_images, batch_size=batch_size, shuffle=True)

autoencoder_model = ViTMaskedAutoencoder().cuda()
autoencoder_filepath = "autoencoder-plain-98-epoch063"
autoencoder_model.load_state_dict(torch.load(autoencoder_filepath, weights_only=True))

loss_fn_l1 = nn.SmoothL1Loss(reduction="none")
loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

optimizer = optim.Adam(autoencoder_model.parameters(), lr=9-4)
optimizer_filepath = "optimizer-plain-98-epoch063"
optimizer.load_state_dict(torch.load(optimizer_filepath))

lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-4)
warmup = 40
num_epochs = 500
epoch_loss = []
validation_loss = []
epoch_loop = tqdm(range(51, num_epochs), total = num_epochs - 50)
for epoch in epoch_loop:
    autoencoder_model.train()
    batch_losses = []
    for n, image_batch in enumerate(tqdm(train_dataloader,total=len(train_dataloader))):

        latent = autoencoder_model.encode(image_batch)
        y_hat = autoencoder_model.decode(latent)
        batch_loss = loss_fn_l1(y_hat, image_batch).mean([-1,-2,-3])
        y_hat = y_hat.repeat(1,3,1,1)
        image_batch = image_batch.repeat(1,3,1,1)
        batch_loss += loss_fn_vgg.forward(y_hat, image_batch).squeeze()
        batch_loss = batch_loss.mean()
        #if (n%25==0): print(batch_loss)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_losses.append(batch_loss.cpu().detach().numpy())            


    epoch_loss.append(np.mean(batch_losses))
    if (epoch > warmup): lr_scheduler.step()
    print(f"\n Average LPIPS + L1 Loss at epoch {epoch}: {epoch_loss[-1]}")
    
    ##Validation
    #This does manual inspection of the reconstructed images
    autoencoder_model.eval()
    val_losses = []
    fig, axs = plt.subplots(2, 8, figsize=(40,10))

    for i, image_idx in enumerate(torch.randint(low=0,  high = len(images_val) - 1, size=(8,))):

        axs[0,i].imshow(images_val[image_idx].numpy(), cmap = "grey")
        axs[0,i].set_title("original")

        
        with torch.no_grad():

            latent = autoencoder_model.encode(images_val[image_idx]
                                                    .unsqueeze(0)
                                                    .unsqueeze(0)
                                                    .to(device))

            y_hat_val = autoencoder_model.decode(latent).squeeze()

            val_loss = loss_fn_l1(y_hat_val, images_val[image_idx].to(device))
            val_losses.append(val_loss.cpu().detach().numpy())
            axs[1,i].imshow(y_hat_val.cpu().detach().numpy(), cmap="grey")
            axs[1,i].set_title("reconstruction")
    
    plt.savefig(f'./reconstructions/epoch-{epoch:03}.png')
    plt.close()
    print(f"\n L1 Validation Loss at epoch {epoch}: {np.mean(val_losses)}")

    if (epoch%3==0):
        torch.save(autoencoder_model.state_dict(), f"./autoencoder-plain-98-epoch{epoch:03}")
        torch.save(optimizer.state_dict(), f"./optimizer-plain-98-epoch{epoch:03}")
