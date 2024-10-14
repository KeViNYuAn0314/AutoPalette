import os
import sys
import numpy as np
import torch
import torch.nn as nn
from fast_pytorch_kmeans import KMeans


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def kmeans_selection(images, model, ipc):
    with torch.no_grad():
        feat = model.embed(images).detach()
        
    index = torch.arange(len(feat),device='cuda')
    kmeans = KMeans(n_clusters=ipc, mode='euclidean')
    kmeans.fit(feat)
    centers = kmeans.centroids
    
    dist_matrix = euclidean_dist(centers, feat)
    idxs = index[torch.argmin(dist_matrix,dim=1)]
    return idxs
    

def kmeans_sample(images, indices_classes, model, n_cls, ipc):
    indices = []
    for c in range(n_cls):
        idx_shuffle = np.random.permutation(indices_classes[c])
        cls_indices = kmeans_selection(images[idx_shuffle], model, ipc).cpu()
        indices.append(idx_shuffle[cls_indices])
    # indices = torch.cat(indices, dim=0)
    indices = np.concatenate(indices, axis=0)
        
    return indices


