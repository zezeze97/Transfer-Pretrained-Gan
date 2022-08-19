import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out

class GeneratorInterpolate(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z1, y, interpolate=False, z2=None):
        assert(z1.size(0) == y.size(0))
        batch_size = z1.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
            if interpolate:
                lam = torch.rand(1).cuda()
                rand_index = torch.randperm(batch_size).cuda()
                y_shuffle = y[rand_index]
                y_shuffle_embed = self.embedding(y_shuffle)
                y_shuffle_embed = y_shuffle_embed / torch.norm(y_shuffle_embed, p=2, dim=1, keepdim=True)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        if interpolate:
            yz = lam * torch.cat([z1, yembed], dim=1) + (1.0 - lam) * torch.cat([z2, y_shuffle_embed], dim=1)
        else:
            yz = torch.cat([z1, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)
        if interpolate:
            return out, y_shuffle, lam
        else:
            return out


class Generator_Small(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 16
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)
        return out



class Generator_Omit_ClassEmbedding(nn.Module):
    def __init__(self, z_dim, nlabels, size, nfilter=64, split=True, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim
        self.split = split
        # Submodules
        self.fc = nn.Linear(z_dim, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)
        if self.split:
            z_dim = int(z.size(1))
            fake_yembed = z[:,z_dim//2:]

            fake_yembed = fake_yembed / torch.norm(fake_yembed, p=2, dim=1, keepdim=True)

            yz = torch.cat([z[:,:z_dim//2], fake_yembed], dim=1)
            out = self.fc(yz)
        else:
            out = self.fc(z)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, extract_feature=False, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        ny = nlabels
        self.extract_feature = extract_feature

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_0_1 = ResnetBlock(1*nf, 2*nf)

        self.resnet_1_0 = ResnetBlock(2*nf, 2*nf)
        self.resnet_1_1 = ResnetBlock(2*nf, 4*nf)

        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_2_1 = ResnetBlock(4*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 8*nf)
        self.resnet_3_1 = ResnetBlock(8*nf, 16*nf)

        self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_4_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_5_1 = ResnetBlock(16*nf, 16*nf)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)


    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        feature = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(feature))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]
        if self.extract_feature:
            return out, feature
        else:
            return out

class DiscriminatorInterpolate(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, extract_feature=False, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        ny = nlabels
        self.extract_feature = extract_feature

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_0_1 = ResnetBlock(1*nf, 2*nf)

        self.resnet_1_0 = ResnetBlock(2*nf, 2*nf)
        self.resnet_1_1 = ResnetBlock(2*nf, 4*nf)

        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_2_1 = ResnetBlock(4*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 8*nf)
        self.resnet_3_1 = ResnetBlock(8*nf, 16*nf)

        self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_4_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_5_1 = ResnetBlock(16*nf, 16*nf)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)


    def forward(self, x, y, interpolate=False, y_shuffle=None, lam=None):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        feature = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(feature))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        output = out[index, y]

        if interpolate:
            out_shffle = out[index, y_shuffle]
            output = lam * output + (1.0 - lam) * out_shffle
            
        if self.extract_feature:
            return output, feature
        else:
            return output


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
