from medmnist import PneumoniaMNIST
import torch
from torch import optim
from torch.autograd.functional import jacobian
from torch.utils.data import Dataset, DataLoader

#import your autoencoder
from vit_autoencoder_model_alpha4 import ViTMaskedAutoencoder

from gamma import Gamma
from metric import RBFMetric
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
img_size=224

data_pneumonia = PneumoniaMNIST(split="train", download=True, size=img_size)
images = torch.tensor(data_pneumonia.imgs, dtype=torch.uint8).to(torch.float16)/255

class CustomImageDataset(Dataset):
    def __init__(self, images):
        #add channel dimension
        self.images = images.unsqueeze(1)
        self.images = self.images.to(device)
        print(f"Dataset shape: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx,:,:,:]

images_dataset = CustomImageDataset(images)
batch_size = 256
dataloader = DataLoader(images_dataset, batch_size=batch_size, shuffle=True)

##############################
##############################
#load autoencoder
# LOAD YOUR OWN AUTOENCODER
###############################

autoencoder = ViTMaskedAutoencoder().cuda()
autoencoder.load_state_dict(torch.load('autoencoder-mir-epoch30', weights_only=True))
autoencoder.eval()

latents = []
print("------------Generating latents------------")
for batch in dataloader:
    latents.append(autoencoder.encode(batch))
latents = torch.cat(latents).to(dtype=torch.float32)
latents_numpy = latents.cpu().detach().numpy()

torch.set_default_dtype(torch.float32)

print("---------Latents generated----------------")
print(f"Generated {latents.shape[0]}latents of dimension {latents.shape[1]}")


#We need a dataset of latents for training the metric
class CustomLatentsDataset(Dataset):
    def __init__(self, latents):
        self.latents = latents

    def __len__(self):
        return len(self.latents)

    def __getitem__(self,idx):
        return latents[idx]

print("--------------Fitting Kmeans---------------")
metric = RBFMetric().cuda()
load_metric = False

#does not work because when I save the model,
# it does not save the variables other than weights
if load_metric:
    metric.load_state_dict("./metric-epoch19")
    
else:
    metric.train_kmeans(latents_numpy)
    print("------------ Training metric --------------")


    latents_dataset = CustomLatentsDataset(latents)
    dataloader = DataLoader(latents_dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(metric.parameters())
    n_epochs_metric = 20
    for epoch in range(n_epochs_metric):
        losses = []
        for batch in dataloader: 
            optimizer.zero_grad()
            m = metric.forward(batch)
            loss = torch.mean((1 - m) ** 2)
            #print(loss)
            losses.append(loss.cpu().detach().numpy())
            loss.backward(retain_graph=True)
            optimizer.step()
            
        print(f"Loss at epoch {epoch}: {np.mean(losses)}")
    torch.save(metric.state_dict(), f"./metric-epoch{epoch}")


#Up to this point we have the metric, it tells us when a point falls near or far of the data
#distribution






#Now we train gamma,
#Gamma is the network that will push the interpolants so that they fall
#close to the data distribution, we use the metric to know that

print("-------------Training gamma------------------")

x0_index = (data_pneumonia.labels == 0).squeeze(1)
x1_index = (data_pneumonia.labels == 1).squeeze(1)
latents_x0_dataset = CustomLatentsDataset(latents[x0_index,:])
max_idx = len(latents_x0_dataset)
x1_latents = latents[x1_index,:]
x1_latents = x1_latents[:max_idx,:]
latents_x1_dataset = CustomLatentsDataset(x1_latents)

batch_size=8
x0_dataloader = DataLoader(latents_x0_dataset, batch_size=batch_size, shuffle=True)
x1_dataloader = DataLoader(latents_x1_dataset, batch_size=batch_size, shuffle=True)


timesteps = torch.linspace(0.0, 1.0, len(latents_x1_dataset)).tolist()

#Gamma model
#the gamma model calculates a push towards the good interpolation path
#the good interpolation path is the one where the metric says it is close
#to the distribution given source, target and timestep
_, latent_dim = latents.shape
gamma = Gamma(latent_dim = latent_dim).to(device=device)


#this calculates the derivative of the gamma model,
#it is needed to compute the conditional flow
def func_jacobian(model, x0, x1, timestep_jac):
    def f(timestep_jac_):
        return model(x0, x1, timestep_jac_)
    return jacobian(f, timestep_jac, create_graph=False, vectorize=True)


optimizer = optim.Adam(gamma.parameters())

n_epochs = 250
losses = []
for epoch in range(n_epochs):
    for i, (x0, x1) in enumerate(zip(x0_dataloader, x1_dataloader)):
        timestep = timesteps[i]

        #first we compute the interpolant at the timestep
        #note that a correction term is added
        gamma_push = gamma(x0,x1,timestep)
        interpolant = timestep*x1 + (1-timestep)*x0 + (timestep*(1-timestep))*gamma_push

        #compute conditional flow
        #this equation is literal what the paper describes
        #you can see how the second and third lines
        #come from the derivative wrt time of the correction term just above
        d_gamma_push = func_jacobian(gamma, x0, x1, timestep)
        conditional_flow = (x1 - x0) / (timesteps[i+1] - timestep) \
                            + (timestep*(1-timestep))* d_gamma_push \
                            + (1 - 2*timestep)*gamma_push

        #measure how deviated from the data is the interpolant
        metric_res = metric.forward(interpolant).unsqueeze(1)

        #compute velocity (is our loss)
        #velocity is the derivative wrt time of the
        velocity = ((conditional_flow**2) * metric_res).sum(dim=-1)
        velocity = velocity.mean()
        #print(f"Velocity: {velocity}")

        optimizer.zero_grad()
        velocity.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(gamma.parameters(), max_norm=1.0)
        optimizer.step()

        velocities.append(velocity.unsqueeze(0))

    loss = torch.cat(velocities)
    losses.append(loss.mean().cpu().detach().numpy())

    print(f"Loss at epoch {epoch}: {losses[-1]}")
    if ((epoch+1)%25 == 0):
        torch.save(gamma.state_dict(), f"./gamma-epoch{epoch:03}")
        torch.save(optimizer.state_dict(), f"./optimizer-gamma-epoch{epoch:03}")

