import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from autoencoder import ViTMaskedAutoencoder
from medmnist import PneumoniaMNIST
from torch.utils.data import Dataset, DataLoader, RandomSampler
import argparse
import lpips
from torch.nn.functional import sigmoid
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='train vit autoencoder')

parser.add_argument('-vgg', action="store_true", default=False)
parser.add_argument('-l1', action = "store_true", default=False)
args = parser.parse_args()
use_vgg = args.vgg
use_l1 = args.l1
print(f"Using vgg perceptual loss: {use_vgg}")
print(f"Using L1 loss: {use_l1}")


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
autoencoder_filepath = "VIT-BCE-VGG-98-epoch054"
autoencoder_model.load_state_dict(torch.load(autoencoder_filepath, weights_only=True))

if use_l1:
    loss_fn_Pixel = nn.SmoothL1Loss()
    pixel_loss_tag = "L1"
else:
    loss_fn_Pixel = nn.BCELoss()
    pixel_loss_tag = "BCE"

if use_vgg:
    loss_fn_VGG = lpips.LPIPS(net="vgg").to(device)
    tag = "VGG"
else:
    tag = "noVGG"

#for validation
loss_fn_FID = FrechetInceptionDistance(feature = 2048, 
                                       normalize=True,
                                       input_img_size=(1,224,224),
                                       antialias=True).cuda()
loss_fn_FID.set_dtype(torch.float64)

optimizer = optim.AdamW(autoencoder_model.parameters(), lr=12e-4)
#optimizer_filepath = "Opt-BCE-VGG-98-epoch054"
#optimizer.load_state_dict(torch.load(optimizer_filepath))


#lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=3e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.65, patience=4, threshold=1, threshold_mode='abs', cooldown=2, min_lr=1e-6, eps=1e-08)
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  [Early Stopping] No improvement... Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0




early_stopping = EarlyStopping()
num_epochs = 100
epoch_loss_Pixel = []
if use_vgg: epoch_loss_VGG = []
validation_epoch_loss = []


#epoch_loss_Pixel = np.load("epoch_loss_pixel-epoch54.npy")
#epoch_loss_VGG = np.load("epoch_loss_VGG-epoch54.npy")
#validation_epoch_loss = np.load("validation_epoch_loss-epoch54.npy")



start_epoch = 1
epoch_loop = tqdm(range(start_epoch, num_epochs), total = num_epochs - start_epoch + 1)
for epoch in epoch_loop:
    autoencoder_model.train()
    batch_losses_Pixel = []
    batch_losses_VGG = []
    for n, image_batch in enumerate(tqdm(train_dataloader,total=len(train_dataloader))):
        
        latent = autoencoder_model.encode(image_batch)
        y_hat = autoencoder_model.decode(latent)
        
        batch_loss_pixel = loss_fn_Pixel(y_hat, image_batch)
        batch_loss_VGG = 0
        if use_vgg:
            y_hat = y_hat.repeat(1,3,1,1)
            image_batch = image_batch.repeat(1,3,1,1)
            batch_loss_VGG = loss_fn_VGG.forward(y_hat, image_batch).mean()
        

        batch_loss = batch_loss_pixel
        if use_vgg:
            batch_loss += batch_loss_VGG

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_losses_Pixel = np.append(batch_losses_Pixel, batch_loss_pixel.cpu().detach().numpy())
        if use_vgg:
            batch_losses_VGG = np.append(batch_losses_VGG, batch_loss_VGG.cpu().detach().numpy())            

    epoch_loss_Pixel = np.append(epoch_loss_Pixel, np.mean(batch_losses_Pixel))
    if use_vgg:
        epoch_loss_VGG = np.append(epoch_loss_VGG, np.mean(batch_losses_VGG))

    print(f"\n Average {pixel_loss_tag} Loss at epoch {epoch}: {epoch_loss_Pixel[-1]}")
    if use_vgg:
        print(f" Average VGG Loss at epoch {epoch}: {epoch_loss_VGG[-1]}")



    #Validation
    #First:
    #This does manual inspection of 8 random reconstructed images
    autoencoder_model.eval()
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
            axs[1,i].imshow(y_hat_val.cpu().detach().numpy(), cmap="grey")
            axs[1,i].set_title("reconstruction")

          
    plt.savefig(f'./reconstructions/{pixel_loss_tag}-{tag}-epoch-{epoch:03}.png')
    plt.close()

    #Second:
    #Calculate validation for whole validation set
    val_batch_loss = []
    for image_batch in tqdm(validation_dataloader, total=len(validation_dataloader)):
        with torch.no_grad():
            latent = autoencoder_model.encode(image_batch)
            y_hat = autoencoder_model.decode(latent)
            
            loss_fn_FID.reset()
            real_imgs = image_batch.repeat(1,3,1,1)
            loss_fn_FID.update(real_imgs, real=True)
            recons_imgs = y_hat.repeat(1,3,1,1)
            loss_fn_FID.update(recons_imgs, real=False)
            val_batch_loss.append(loss_fn_FID.compute().cpu().numpy())

    validation_epoch_loss = np.append(validation_epoch_loss, np.mean(val_batch_loss))

    lr_scheduler.step(validation_epoch_loss[-1])

    print(f"\nValidation Loss at epoch {epoch}: {validation_epoch_loss[-1]}")
    print(f"Learning rate: {lr_scheduler.get_last_lr()}")

    #save every 3 epochs and plot losses (overwrite)
    if (epoch%3==0):
        torch.save(autoencoder_model.state_dict(), f"./VIT-{pixel_loss_tag}-{tag}-98-epoch{epoch:03}")
        torch.save(optimizer.state_dict(), f"./Opt-{pixel_loss_tag}-{tag}-98-epoch{epoch:03}")

        np.save(f"epoch_loss_pixel-epoch{epoch}", epoch_loss_Pixel)
        if use_vgg: np.save(f"epoch_loss_VGG-epoch{epoch}", epoch_loss_VGG)
        np.save(f"validation_epoch_loss-epoch{epoch}", validation_epoch_loss)


    #Save plot figures for train and validation losses
    final_epoch=epoch
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].plot(np.arange(final_epoch), np.squeeze(epoch_loss_Pixel), color="blue")
    axs[1].plot(np.arange(final_epoch), np.squeeze(validation_epoch_loss), color="blue")
    axs[1].legend(["val loss (FID)"])
    if (use_vgg):
        axs[0].plot(np.arange(final_epoch), np.squeeze(epoch_loss_VGG), color="red")
        axs[0].plot(np.arange(final_epoch), np.squeeze(epoch_loss_Pixel + epoch_loss_VGG), color="green")
        axs[0].legend([pixel_loss_tag, 'vgg loss', 'total loss'])

    else:
        axs[0].legend([pixel_loss_tag])

    plt.savefig(f'loss plot-{pixel_loss_tag}-{tag}.png')
    plt.close()

    early_stopping(validation_epoch_loss[-1])
    if early_stopping.early_stop:
        print(" Early Stopping..")
        torch.save(autoencoder_model.state_dict(), f"./VIT-{pixel_loss_tag}-{tag}-98-epoch{epoch:03}")
        torch.save(optimizer.state_dict(), f"./Opt-{pixel_loss_tag}-{tag}-98-epoch{epoch:03}")
        np.save(f"epoch_loss_pixel-epoch{epoch}", epoch_loss_Pixel)
        if use_vgg: np.save(f"epoch_loss_VGG-epoch{epoch}", epoch_loss_VGG)
        np.save(f"validation_epoch_loss-epoch{epoch}", validation_epoch_loss)
        break