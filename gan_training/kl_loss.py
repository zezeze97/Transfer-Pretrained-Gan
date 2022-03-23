import torch 



def kl_divergence(mu, cov, multivariate_normal_dim):
    kl_loss = 0.5*(torch.linalg.norm(mu) - cov.diag().log().sum() + cov.trace() - multivariate_normal_dim)
    return kl_loss


