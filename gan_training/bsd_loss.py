import torch
import torch.nn as nn


class BSD_loss(nn.Module):
    '''
    Input: Feature (B, )
    '''
    def __init__(self, num_of_index):
        super(BSD_loss, self).__init__()
        self.num_of_index = num_of_index

    def forward(self, feature):
        u,s,v = torch.svd(feature)
        bsd_loss = 1.0 / torch.mean(torch.pow(s[-self.num_of_index:],2)).clamp_(1e-6)
        return bsd_loss