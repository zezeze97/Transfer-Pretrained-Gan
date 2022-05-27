import torch
from torch import distributions 
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def get_zdist(dist_name, dim, mean=None, cov=None, gmm_components_weight=None, gmm_mean=None, gmm_cov=None, latentvecs=None, device=None):
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
    elif dist_name == 'kde':
        zdist = KDE(latentvecs, device)
    elif dist_name == 'gmm2gauss':
        zdist = GMM2Gauss(dim, gmm_components_weight, gmm_mean, gmm_cov, device)
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


class GMM2Gauss:  
   def __init__(self, dim=None, gmm_components_weight=None, gmm_mean=None, gmm_cov=None, device=None):  
       self.gmm_components_weight = gmm_components_weight  
       self.gmm_mean = gmm_mean
       self.gmm_cov = gmm_cov
       self.device = device
       self.dim = dim
   def sample(self, sample_shape, use_gmm):
        num_sample = sample_shape[0]
        if use_gmm:
            num_for_classes = np.random.multinomial(n=num_sample, pvals=self.gmm_components_weight)
            points = []  
            for component_index, num in enumerate(num_for_classes):  
                mean = self.gmm_mean[component_index,:]
                cov = self.gmm_cov[component_index,:,:]
                points.append(np.random.multivariate_normal(mean=mean, cov=cov,size=num))  
            output = torch.FloatTensor(np.concatenate(points)).to(self.device) 
        else:
            sample = np.random.randn(num_sample, self.dim)
            output = torch.FloatTensor(sample).to(self.device) 
        return output


class KDE:  
    def __init__(self, latentvecs=None, device=None): 
        print('fitting latentvecs using kde...') 
        # use grid search cross-validation to optimize the bandwidth
        params = {"bandwidth": np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(latentvecs)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        # use the best estimator to compute the kernel density estimate
        self.kde = grid.best_estimator_
        self.device = device
 
    def sample(self, sample_shape):  
        num_sample = sample_shape[0]
        samles = self.kde.sample(n_samples=num_sample)
        return torch.FloatTensor(samles).to(self.device) 



if __name__ == '__main__':
    gmm_components_weight = np.load('/home/zhangzr/GAN_stability/output/vec2img/pets_256dim_special_init_fix/gmm_components_weights.npy')
    gmm_mean = np.load('/home/zhangzr/GAN_stability/output/vec2img/pets_256dim_special_init_fix/gmm_mean.npy')
    gmm_cov = np.load('/home/zhangzr/GAN_stability/output/vec2img/pets_256dim_special_init_fix/gmm_cov.npy')
    zdist = get_zdist(dist_name='gmm2gauss', dim=256, gmm_components_weight=gmm_components_weight, gmm_mean=gmm_mean, gmm_cov=gmm_cov, device=None)
    sample = zdist.sample((16,),use_gmm=False)
    print('sample shape: ', sample.shape)