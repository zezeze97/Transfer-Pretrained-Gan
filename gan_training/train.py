# coding: utf-8
import imp
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
from gan_training.mmd_loss import MMD_loss
from gan_training.bss_loss import BSS_loss

class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, frozen_generator=False, frozen_discriminator=False, 
                 frozen_generator_param_list=None, frozen_discriminator_param_list=None):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.frozen_generator = frozen_generator
        self.frozen_discriminator = frozen_discriminator
        self.frozen_generator_param_list = frozen_generator_param_list
        self.frozen_discriminator_param_list = frozen_discriminator_param_list

    def generator_trainstep(self, y, z):
        assert(y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        if self.frozen_generator:
            toggle_grad(self.generator, False, parameters_list=self.frozen_generator_param_list)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item()

    def discriminator_trainstep(self, x_real, y, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        if self.frozen_discriminator:
            toggle_grad(self.discriminator, False, parameters_list=self.frozen_discriminator_param_list)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

class TrainerClassInterpolate(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, frozen_generator=False, frozen_discriminator=False, 
                 frozen_generator_param_list=None, frozen_discriminator_param_list=None, mix_prob=0.5, contract_lam=0.1):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.frozen_generator = frozen_generator
        self.frozen_discriminator = frozen_discriminator
        self.frozen_generator_param_list = frozen_generator_param_list
        self.frozen_discriminator_param_list = frozen_discriminator_param_list
        self.mix_prob = torch.tensor(mix_prob)
        self.contract_lam = contract_lam
        self.l2loss = torch.nn.MSELoss()

    def generator_trainstep(self, y, z):
        assert(y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        if self.frozen_generator:
            toggle_grad(self.generator, False, parameters_list=self.frozen_generator_param_list)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()
        temp_prob = torch.rand(1)
        if temp_prob < self.mix_prob:
            x_mix, rand_index, lam = self.generator(z, y, interpolate=True)
            d_fake, _ = self.discriminator(x_mix, y)
        else:
            x_fake = self.generator(z, y, interpolate=False)
            d_fake, _ = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()
        self.g_optimizer.step()

        return gloss.item()

    def discriminator_trainstep(self, x_real, y, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        if self.frozen_discriminator:
            toggle_grad(self.discriminator, False, parameters_list=self.frozen_discriminator_param_list)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real, _ = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        temp_prob = torch.rand(1)
        with torch.no_grad():
            if temp_prob < self.mix_prob:
                x_fake, rand_index, lam = self.generator(z, y, interpolate=True)
                x_fake1 = self.generator(z, y)
                _, d_fake_feature1 = self.discriminator(x_fake1, y)
                d_fake_feature2 = d_fake_feature1[rand_index,:]
            else:
                x_fake = self.generator(z, y, interpolate=False)
        x_fake.requires_grad_()
        if temp_prob < self.mix_prob:
            d_fake, d_fake_mix_feature = self.discriminator(x_fake, y)
            dloss_fake = self.compute_loss(d_fake, 0)
            dloss_contract = self.l2loss(d_fake_mix_feature, lam * d_fake_feature1 + (1.0 - lam) * d_fake_feature2)
            dloss_fake = dloss_fake + self.contract_lam * dloss_contract
        else:
            d_fake, _ = self.discriminator(x_fake, y)
            dloss_fake = self.compute_loss(d_fake, 0)
            dloss_contract = torch.tensor(0.)


        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)
    
        return dloss.item(), reg.item(), dloss_contract.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out, _ = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg


class Learnable_GMM_Trainer(object):
    def __init__(self, gmm_layer, generator, discriminator, g_optimizer, d_optimizer, gmm_optimizer,
                 gan_type, reg_type, reg_param, frozen_generator=False, frozen_discriminator=False, 
                 frozen_generator_param_list=None, frozen_discriminator_param_list=None):
        self.gmm_layer = gmm_layer
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gmm_optimizer = gmm_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.frozen_generator = frozen_generator
        self.frozen_discriminator = frozen_discriminator
        self.frozen_generator_param_list = frozen_generator_param_list
        self.frozen_discriminator_param_list = frozen_discriminator_param_list
        

    def generator_trainstep(self, y, frozen_gmm_layer=False):
        if frozen_gmm_layer:
            toggle_grad(self.gmm_layer, False)
        else:
            toggle_grad(self.gmm_layer, True)
        toggle_grad(self.generator, True)
        if self.frozen_generator:
            toggle_grad(self.generator, False, parameters_list=self.frozen_generator_param_list)
        toggle_grad(self.discriminator, False)
        self.gmm_layer.train()
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()
        if not frozen_gmm_layer:
            self.gmm_optimizer.zero_grad()
        if frozen_gmm_layer:
            with torch.no_grad():
                z = self.gmm_layer(y.size(0))
        else:
            z = self.gmm_layer(y.size(0))
        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()
        if not frozen_gmm_layer:
            self.gmm_optimizer.step()


        return gloss.item()

    def discriminator_trainstep(self, x_real, y):
        toggle_grad(self.gmm_layer, False)
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        if self.frozen_discriminator:
            toggle_grad(self.discriminator, False, parameters_list=self.frozen_discriminator_param_list)
        self.gmm_layer.train()
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            z = self.gmm_layer(y.size(0))
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg



class Trainer_Limited_Data(object):
    def __init__(self, generator, aux_generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, frozen_generator=False, frozen_discriminator=False, 
                 frozen_generator_param_list=None, frozen_discriminator_param_list=None, mmd_loss_lambda=None, aux_real_lambda=None):
        self.generator = generator
        self.aux_generator = aux_generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.frozen_generator = frozen_generator
        self.frozen_discriminator = frozen_discriminator
        self.frozen_generator_param_list = frozen_generator_param_list
        self.frozen_discriminator_param_list = frozen_discriminator_param_list
        self.mmd_loss = MMD_loss()
        self.mmd_loss_lambda = mmd_loss_lambda
        self.aux_real_lambda = aux_real_lambda

    def generator_trainstep(self, x_real, y, z):
        assert(y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        if self.frozen_generator:
            toggle_grad(self.generator, False, parameters_list=self.frozen_generator_param_list)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        d_fake, fake_feature = self.discriminator(x_fake, y)
        d_real, real_feature = self.discriminator(x_real, y)
        g_mmd_loss = self.compute_loss(d_out=None, target=None, fake_feature=fake_feature, real_feature=real_feature, loss_type='mmd')
        g_classification_loss = self.compute_loss(d_out=d_fake, target=1, fake_feature=None, real_feature=None, loss_type=None)
        gloss = g_classification_loss + self.mmd_loss_lambda * g_mmd_loss
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item(), g_classification_loss.item(), g_mmd_loss.item()

    def discriminator_trainstep(self, x_real, y, z, z_aux):
        toggle_grad(self.generator, False)
        toggle_grad(self.aux_generator, False)
        toggle_grad(self.discriminator, True)
        if self.frozen_discriminator:
            toggle_grad(self.discriminator, False, parameters_list=self.frozen_discriminator_param_list)
        self.generator.train()
        self.discriminator.train()
        self.aux_generator.eval()
        self.d_optimizer.zero_grad()

        
        # generate x_aux
        #with torch.no_grad():
        x_aux = self.aux_generator(z_aux, y)
        # On aux and real data
        x_real.requires_grad_()
        x_aux.requires_grad_()

        d_real, real_feature = self.discriminator(x_real, y)
        d_aux, aux_feature = self.discriminator(x_aux, y)
        dloss_real = self.compute_loss(d_out=d_real, target=1, fake_feature=None, real_feature=None, loss_type=None)
        dloss_aux = self.compute_loss(d_out=d_aux, target=1, fake_feature=None, real_feature=None, loss_type=None)

        dloss_aux_real = dloss_real + self.aux_real_lambda * dloss_aux
        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_aux_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean() # + self.aux_real_lambda * self.reg_param * compute_grad2(d_aux, x_real).mean()
            reg.backward()
        else:
            dloss_aux_real.backward()
        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake, fake_feature = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_out=d_fake, target=0, fake_feature=None, real_feature=None, loss_type=None)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_aux_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)
        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target, fake_feature=None, real_feature=None, loss_type=None):
        if loss_type is None:
            targets = d_out.new_full(size=d_out.size(), fill_value=target)
            if self.gan_type == 'standard':
                loss = F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.gan_type == 'wgan':
                loss = (2*target - 1) * d_out.mean()
            else:
                raise NotImplementedError
        if loss_type == 'mmd':
            loss = self.mmd_loss(fake_feature, real_feature)
        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out, _ = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

class Trainer_BSS(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, frozen_generator=False, frozen_discriminator=False, 
                 frozen_generator_param_list=None, frozen_discriminator_param_list=None, bss_loss_lambda=None, bss_num_of_index=None):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.frozen_generator = frozen_generator
        self.frozen_discriminator = frozen_discriminator
        self.frozen_generator_param_list = frozen_generator_param_list
        self.frozen_discriminator_param_list = frozen_discriminator_param_list
        self.bss_loss_fun = BSS_loss(num_of_index=bss_num_of_index)
        self.bss_loss_lambda = bss_loss_lambda

    def generator_trainstep(self, y, z):
        assert(y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        if self.frozen_generator:
            toggle_grad(self.generator, False, parameters_list=self.frozen_generator_param_list)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        d_fake, d_fake_feature = self.discriminator(x_fake, y)
        cls_loss = self.compute_loss(d_fake, 1)
        bss_loss = self.bss_loss_fun(d_fake_feature)
        gloss = cls_loss + self.bss_loss_lambda * bss_loss
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item(), cls_loss.item(), bss_loss.item()

    def discriminator_trainstep(self, x_real, y, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        if self.frozen_discriminator:
            toggle_grad(self.discriminator, False, parameters_list=self.frozen_discriminator_param_list)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real, d_real_feature = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake, d_fake_feature = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_fake, 0) + self.bss_loss_lambda * self.bss_loss_fun(d_fake_feature)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()


    def compute_loss(self, d_out, target, feature=None, loss_type=None):
        if loss_type is None:
            targets = d_out.new_full(size=d_out.size(), fill_value=target)
            if self.gan_type == 'standard':
                loss = F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.gan_type == 'wgan':
                loss = (2*target - 1) * d_out.mean()
            else:
                raise NotImplementedError
        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out, _ = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg



# Utility functions
def toggle_grad(model, requires_grad, parameters_list=None):
    if parameters_list is None:
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    else:
        for k,v in model.named_parameters():
            if k in parameters_list:
                v.requires_grad_(requires_grad)
            


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
