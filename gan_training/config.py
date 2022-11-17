import yaml
from torch import optim
from os import path
from gan_training.models import generator_dict, discriminator_dict
from gan_training.train import toggle_grad


# General config
def load_config(path, default_path):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def build_models(config):
    # Get classes
    Generator = generator_dict[config['generator']['name']]
    Discriminator = discriminator_dict[config['discriminator']['name']]

    # Build models
    generator = Generator(
        z_dim=config['z_dist']['dim'],
        nlabels=config['data']['nlabels'],
        size=config['data']['img_size'],
        **config['generator']['kwargs']
    )
    discriminator = Discriminator(
        config['discriminator']['name'],
        nlabels=config['data']['nlabels'],
        size=config['data']['img_size'],
        **config['discriminator']['kwargs']
    )

    return generator, discriminator

def build_generator(config):
    # Get classes
    Generator = generator_dict[config['generator']['name']]
    # Build models
    generator = Generator(
        z_dim=config['z_dist']['dim'],
        nlabels=config['data']['nlabels'],
        size=config['data']['img_size'],
        **config['generator']['kwargs']
    )
    
    return generator



def build_optimizers(generator, discriminator, config):
    optimizer = config['training']['optimizer']
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    equalize_lr = config['training']['equalize_lr']
    frozen_generator = config['training']['frozen_generator']
    frozen_discriminator = config['training']['frozen_discriminator']
    
    toggle_grad(generator, True)
    toggle_grad(discriminator, True)

    if frozen_generator:
        toggle_grad(generator, False, parameters_list=config['training']['frozen_generator_param_list'])
    
    if frozen_discriminator:
         toggle_grad(discriminator, False, parameters_list=config['training']['frozen_discriminator_param_list'])

    if equalize_lr:
        g_gradient_scales = getattr(generator, 'gradient_scales', dict())
        d_gradient_scales = getattr(discriminator, 'gradient_scales', dict())

        g_params = get_parameter_groups(generator.parameters(),
                                        g_gradient_scales,
                                        base_lr=lr_g)
        d_params = get_parameter_groups(discriminator.parameters(),
                                        d_gradient_scales,
                                        base_lr=lr_d)
    else:
        if frozen_generator:
            g_params = filter(lambda p: p.requires_grad, generator.parameters())
        else:
            g_params = generator.parameters()

        if frozen_discriminator:
            d_params = filter(lambda p: p.requires_grad, discriminator.parameters())
        else:
            d_params = discriminator.parameters()

    # Print learnable param
    learnable_generator_param_name = []
    for k,v in generator.named_parameters():
        if v.requires_grad:
            learnable_generator_param_name.append(k)
    learnable_discriminator_param_name = []
    for k,v in discriminator.named_parameters():
        if v.requires_grad:
            learnable_discriminator_param_name.append(k)
    print('Learnable Generator Layers: ',learnable_generator_param_name)
    print('Learnable Discriminator Layers: ',learnable_discriminator_param_name)

    # Optimizers
    if optimizer == 'rmsprop':
        g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0., 0.99), eps=1e-8)
        d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(0., 0.99), eps=1e-8)
    elif optimizer == 'sgd':
        g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
        d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)

    

    return g_optimizer, d_optimizer


def build_lr_scheduler(optimizer, config, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_anneal_every'],
        gamma=config['training']['lr_anneal'],
        last_epoch=last_epoch
    )
    return lr_scheduler


# Some utility functions
def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({
            'params': [p],
            'lr': c * base_lr
        })
    return param_groups
