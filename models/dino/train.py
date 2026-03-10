import torch
from medmnist import PneumoniaMNIST
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from DINO_Classifier import DINO_Classifier
from torch.optim import Adam
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#data
img_size=224
#train
data_pneumonia_train = PneumoniaMNIST(split="train", download=True, size=img_size)
#validation
data_pneumonia_val = PneumoniaMNIST(split="val", download=True, size=img_size)

class CustomImageDataset(Dataset):
    def __init__(self, data):
        #add channel dimension
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.images = torch.tensor(data.imgs, dtype=torch.uint8).unsqueeze(1).to(device)
        self.images = processor(images=self.images, return_tensors='pt')['pixel_values']
        self.labels = torch.tensor(data.labels, dtype=torch.float).squeeze().to(device)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_ds = CustomImageDataset(data_pneumonia_train)
val_ds = CustomImageDataset(data_pneumonia_val)
train_dl = DataLoader(train_ds, batch_size=24, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=24, shuffle=True)


dino = DINO_Classifier().to(device)
optimizer = Adam(dino.head.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
metric = BinaryAccuracy().to(device)

#Training loop



num_epochs = 100
epoch_loss = []
accuracies = []
best_accuracy = 0
early_stop_counter = 0
patience = 5
for epoch in tqdm(range(1, num_epochs+1)):
    batch_losses = []
    for imgs, labels in tqdm(train_dl):
        preds = dino(imgs)
        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_loss.append(np.mean(batch_losses))

    metric.reset()
    for imgs, labels in tqdm(val_dl):
        with torch.no_grad():
            preds = dino(imgs)
        metric(preds, labels) 
    accuracy = metric.compute()
    accuracies.append(accuracy.cpu().numpy())
    print(f"Validation Accuracy for epoch {epoch}: {accuracy}")

    #Save plot figures for train and validation
    final_epoch=epoch
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].plot(np.arange(final_epoch), np.squeeze(epoch_loss), color="blue")
    axs[1].plot(np.arange(final_epoch), accuracies, color="blue")
    axs[0].legend(["Binary Cross Entropy Loss"])
    axs[1].legend(["validation accuracy"])

    plt.savefig(f'loss plot.png')
    plt.close()

    #Save best and early stop at accuracy over 99
    if (accuracies[-1] >= 0.99):
        torch.save(dino.state_dict(), f"./Dino-classifier-99acc-epoch{epoch:03}")
        break

    elif (accuracies[-1] >= best_accuracy + 0.002):
        torch.save(dino.state_dict(), f"./Dino-classifier-epoch{epoch:03}")
        best_accuracy = accuracies[-1]
        early_stop_counter = 0

    else:
        early_stop_counter += 1
        if patience == early_stop_counter: break

