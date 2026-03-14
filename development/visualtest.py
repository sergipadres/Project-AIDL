import torch
from medmnist import PneumoniaMNIST
from vit_autoencoder_model_alpha4 import ViTMaskedAutoencoder
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float16)


autoencoder_filepath = './autoencoder-mir-epoch9'

autoencoder = ViTMaskedAutoencoder().to(device)
autoencoder.load_state_dict(torch.load(autoencoder_filepath, weights_only=True))
autoencoder.eval()
img_size=224
data_pneumonia_val = PneumoniaMNIST(split="val", download=True, size=img_size)
images = torch.tensor(data_pneumonia_val.imgs, dtype=torch.uint8).to(dtype=torch.float16)/255



#print 5 random images and its reconstructions
fix, axs = plt.subplots(2, 5, figsize=(20,5))
for i, image_idx in enumerate(torch.randint(low=0,  high = len(images) - 1, size=(5,))):
    #draw original
    axs[0,i].imshow(images[image_idx].numpy(), cmap = "grey")
    axs[0,i].set_title("Original")

    with torch.no_grad():
    #reconstruct with autoencoder and draw
        z = autoencoder.encode(images[image_idx].unsqueeze(0).unsqueeze(0).to(device))
        reconstruction = autoencoder.decode(z)
        axs[1,i].imshow(reconstruction.squeeze().cpu().numpy() ,cmap="grey")
        axs[1,i].set_title("reconstruction")

plt.show()
