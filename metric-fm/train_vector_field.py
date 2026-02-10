
import torch
from torch import nn, optim
from gamma import Gamma
from vector_field import VectorField
from medmnist import PneumoniaMNIST
import torch.utils.data as data
import numpy as np

#data
img_size=224

data_pneumonia = PneumoniaMNIST(split="train", download=True, size=img_size)
print(data_pneumonia)
healthy_indices = (data_pneumonia.labels == 0).squeeze()
sick_indices = (data_pneumonia.labels == 1).squeeze()
print(data_pneumonia.imgs.shape)
healthy_images = torch.tensor(data_pneumonia.imgs[healthy_indices,:,:], dtype=torch.uint8).float()/255
sick_images = torch.tensor(data_pneumonia.imgs[sick_indices,:,:], dtype=torch.uint8).float()/255
max_idx = len(healthy_images)
sick_images = sick_images[:max_idx,:,:]

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

healthy_images_ds = CustomImageDataset(healthy_images)
sick_images_ds = CustomImageDataset(sick_images)

batch_size=10
healthy_dl = DataLoader(healthy_images_ds, batch_size=batch_size, shuffle=True)
sick_dl = DataLoader(sick_images_ds, batch_size=batch_size, shuffle=True)





#############
#############
#LOAD YOUR OWN AUTOENCODER
#############
#############

autoencoder = ViTMaskedAutoencoder().cuda()
autoencoder.load_state_dict(torch.load('YOUR_PATH', weights_only=True))
autoencoder.eval()




#Generate healthy latents
healthy_latents = []
for batch in healthy_dl:
    healthy_latents.append(autoencoder.encode(batch))

sick_latents = []
for batch in sick_dl:
    sick_latents.append(autoencoder.encode(batch))

#################
#################
# USE GAMMA WITH THE SAME
# LATENT DIMENSIONALITY
# AS YOUR AUTOENCODER
#################
#################


gamma = gamma(latent_dim = 98)
gamma.load_state_dict(torch.load("YOUR_PATH"), weights_only=True)
gamma.eval()

vector_field = VectorField(latent_dim = 98)
vector_field.train()

#loss
loss_fn = nn.MSELoss()

#optimizer
optimizer = optim.Adam(vector_field_model.parameters(), lr=0.001)

#as much timesteps as images
#for bigger dataset more timesteps
timesteps = torch.linspace(0.0, 1.0, len(healthy_images)).tolist()


def func_jacobian(model, x0, x1, timestep_jac):
    def f(timestep_jac_):
        return model(x0, x1, timestep_jac_)
    return jacobian(f, timestep_jac, create_graph=False, vectorize=True)



epochs = 100

for epoch in range(epochs):
    interpolants, conditional_flows = [], []
    for i, (x0, x1) in enumerate(zip(healthy_latents, sick_latents)):

        timestep = timesteps[i]
        gamma_push = gamma(x0,x1, timestep)

        interpolant = timestep*x1 + (1-timestep)*x0 + (timestep*(1-timestep))*gamma_push
        interpolants.append(interpolant)

        d_gamma_push = func_jacobian(gamma, x0, x1, timestep)

        conditional_flow = (x1 - x0) / (timesteps[i+1] - timestep) \
                                + (timestep*(1-timestep))* d_gamma_push \
                                + (1 - 2*timestep)*gamma_push

        conditional_flows.append(conditional_flow)

    losses = []
    for (interpolant, timestep, conditional_flow) in zip(interpolants, timesteps, conditional_flows):
        
        vector_field = vector_field_model(interpolant, timestep)
        loss = loss_fn(vector_field, conditional_flow)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    meanloss = torch.mean(torch.tensor(losses))
    print(f"Loss for epoch {epoch}: {meanloss}")





