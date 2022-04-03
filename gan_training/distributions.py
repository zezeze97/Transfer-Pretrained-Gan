import torch
from torch import distributions 
import numpy as np


def get_zdist(dist_name, dim, mean=None, cov=None, gmm_components_weight=None, gmm_mean=None, gmm_cov=None, device=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    elif dist_name == 'multivariate_normal' :
        mean.to(device)
        cov.to(device)
        zdist = distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
    elif dist_name == 'gmm':
        zdist = GMM(gmm_components_weight, gmm_mean, gmm_cov, device)

    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def get_ydist(nlabels, device=None):
    logits = torch.zeros(nlabels, device=device)
    ydist = distributions.categorical.Categorical(logits=logits)

    # Add nlabels attribute
    ydist.nlabels = nlabels

    return ydist


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2

    return z

class GMM:  
   def __init__(self, gmm_components_weight=None, gmm_mean=None, gmm_cov=None, device=None):  
       self.gmm_components_weight = gmm_components_weight  
       self.gmm_mean = gmm_mean
       self.gmm_cov = gmm_cov
       self.device = device
 
   def sample(self, sample_shape):  
       num_sample = sample_shape[0]
       num_for_classes = np.random.multinomial(n=num_sample, pvals=self.gmm_components_weight)
       points = []  
       for component_index, num in enumerate(num_for_classes):  
           mean = self.gmm_mean[component_index,:]
           cov = self.gmm_cov[component_index,:,:]
           points.append(np.random.multivariate_normal(mean=mean, cov=cov,size=num))  
       return torch.FloatTensor(np.concatenate(points)).to(self.device)  
