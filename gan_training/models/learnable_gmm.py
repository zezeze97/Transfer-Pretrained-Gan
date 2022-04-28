import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import distributions 
import numpy as np

class GMM_Layer(nn.Module):
    def __init__(self, z_dim, n_components, gmm_components_weights, device='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.n_components = n_components
        self.gmm_components_weights = gmm_components_weights
        self.gmm_layer = nn.ModuleList([nn.Linear(self.z_dim, self.z_dim, device=device) for i in range(self.n_components)])
        self.standard_gauss = distributions.Normal(loc=torch.zeros(self.z_dim, device=device), scale=torch.ones(self.z_dim, device=device))
        

    def forward(self, num_sample):
        num_for_classes = np.random.multinomial(n=num_sample, pvals=self.gmm_components_weights)
        output_list = []
        for i,num_of_class in enumerate(num_for_classes):
            if num_of_class > 0:
                standard_gauss = self.standard_gauss.sample((num_of_class,))
                shift_gauss = self.gmm_layer[i](standard_gauss)
                output_list.append(shift_gauss)
        
        output = torch.cat(output_list, dim=0)

        return output


if __name__ == '__main__':
    
    device = 'cuda:0'
    # device = 'cpu'
    gmm_components_weights = np.load('output/vec2img/cars_256dim/gmm_components_weights.npy')
    net = GMM_Layer(z_dim=256, n_components=5, gmm_components_weights=gmm_components_weights, device=device)
    for k,v in net.named_parameters():
        print(k,v.shape)
    output = net(32)
    print(output.shape)
        