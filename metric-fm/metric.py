import torch
from torch import nn
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
torch.set_default_dtype(torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RBFMetric(nn.Module):
    def __init__(self,
                 n_clusters = 100,
                 kappa = 1,):
        super().__init__()
        self.n_clusters = n_clusters
        self.kappa = kappa
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        Wdata = (torch.rand(self.n_clusters, 1)**2).to(torch.float16)
        self.W = nn.Parameter(Wdata, requires_grad=True)

    def compute_cluster_points_indexes(self):
        cluster_points_indexes =  [self.kmeans.labels_[self.kmeans.labels_ == k]
                                        for k in range(self.n_clusters)]
        return cluster_points_indexes

    def compute_lambdas(self,data):
        data = torch.tensor(data)
        sum_k = torch.tensor([torch.sum(torch.tensor([torch.linalg.norm(point - self.cluster_centers[k])
                  for point in data[self.cluster_points_indexes[k],:]]))
                     for k in range(self.n_clusters)])
        lambda_k = torch.tensor([torch.pow(self.kappa / self.cluster_sizes[k] * sum_k[k], -2)*0.5
                    for k in range(self.n_clusters)])
        return lambda_k

    def train_kmeans(self, data: np.array):
        print(f"Fitting kmeans")
        self.kmeans.fit(data)
        print(f"Computing auxiliary variables")
        self.cluster_centers = torch.tensor(self.kmeans.cluster_centers_ , requires_grad=False)
        self.cluster_sizes = Counter(self.kmeans.labels_)
        self.cluster_points_indexes = self.compute_cluster_points_indexes()
        self.lambdas = self.compute_lambdas(data)
        self.cluster_centers = self.cluster_centers.to(torch.device("cuda"))
        print(f"done")

    def forward(self, point):

        with torch.no_grad():
            self.W.data.clamp(min=0.001)

        phi_k = torch.stack([torch.exp(-0.5 * self.lambdas[k] * torch.linalg.norm(point - self.cluster_centers[k], dim=1)**2)
                                for k in range(self.n_clusters)])

        metric = torch.mul(self.W, phi_k)

        metric = torch.sum(metric, dim=0) #(batch_size)

        #We need this following line to output:
        #  1 close to data
        #  0 far from data
        metric = 1 - torch.abs(1 - metric)

        return metric

    def clampW(self):
        self.W = torch.clamp(self.W, min=0.00001).to(torch.float16)
