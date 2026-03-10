import torch
from torch import nn
from transformers import AutoModel

class DINO_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_backbone = AutoModel.from_pretrained('facebook/dinov2-base')
        self.dino_backbone.eval()
        self.head = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dino_backbone(x)["last_hidden_state"][:,0,:].squeeze()
        x = self.head(x)
        x = self.sigmoid(x)
        return x.squeeze()


