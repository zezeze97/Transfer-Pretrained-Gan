import torch 



def kl_divergence(mu, cov, multivariate_normal_dim, eps = 0.0001):
    
    det_cov = torch.det(cov)
    eps = torch.tensor(eps)
    if torch.abs(det_cov) < eps:
        kl_loss = 0.5*(torch.linalg.norm(mu) - torch.log(eps) + torch.trace(cov) - multivariate_normal_dim)
    else:
        kl_loss = 0.5*(torch.linalg.norm(mu) - torch.log(det_cov) + torch.trace(cov) - multivariate_normal_dim)
    return kl_loss



