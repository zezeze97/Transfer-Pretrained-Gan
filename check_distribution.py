import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.config import load_config, build_generator
from gan_training.distributions import get_zdist, get_ydist
from gan_training.checkpoints import CheckpointIO
from collections import OrderedDict
import torchvision
import numpy as np
from tqdm import tqdm


def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename


# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--num_of_batches', type=int, help='Number of batch fake image.')
parser.add_argument('--outdir', type=str, help='Save path.')
args = parser.parse_args()

# config 
config = load_config(args.config, default_path='configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)


# Create missing directories
out_dir = args.outdir
if not path.exists(out_dir):
    os.makedirs(out_dir)

# use gpu
device = torch.device("cuda:0" if is_cuda else "cpu")


# Number of labels
nlabels = config['data']['nlabels']


# Create models
generator = build_generator(config)
print(generator)



# Put models on gpu if needed
generator = generator.to(device)


# Load pretrained ckpt
finetune_mode = config['training']['finetune']
change_generator_embedding_layer = config['training']['change_generator_embedding_layer']
change_discriminator_fc_layer = config['training']['change_discriminator_fc_layer']
if finetune_mode:
    if change_generator_embedding_layer and change_discriminator_fc_layer:
        print('change generator embedding layer!!!')
        if config['training']['pretrain_ckpt_file'] is None:
            # load pretrained generator
            generator.load_state_dict(torch.load(config['training']['generator_pretrained_ckpt_file']))
            print('pretrained generator loaded!')
        else:
            # load pretrained generator
            pretrained_generator_loaded_dict = remove_module_str_in_state_dict(torch.load(config['training']['pretrain_ckpt_file'])['generator'])
            generator_state_dict = generator.state_dict()
            new_dict = {k: v for k, v in pretrained_generator_loaded_dict.items() if k != 'embedding.weight'}
            generator_state_dict.update(new_dict)
            generator.load_state_dict(generator_state_dict)
            print('pretrained generator loaded!')
    else:
        pretrained_generator_loaded_dict = remove_module_str_in_state_dict(torch.load(config['training']['pretrain_ckpt_file'])['generator']) 
        generator.load_state_dict(pretrained_generator_loaded_dict)
        print('pretrained generator loaded!')




# load mean and cov of latentvecs if neccessary
if config['z_dist']['type'] == 'multivariate_normal':
    mean_path = config['z_dist']['mean_path']
    cov_path = config['z_dist']['cov_path']
    mean = torch.FloatTensor(np.load(mean_path))
    cov = torch.FloatTensor(np.load(cov_path))
# load gmm parameters if neccessary
if config['z_dist']['type'] == 'gmm':
    gmm_components_weight = np.load(config['z_dist']['gmm_components_weight'])
    gmm_mean = np.load(config['z_dist']['gmm_mean'])
    gmm_cov = np.load(config['z_dist']['gmm_cov'])


# Distributions
ydist = get_ydist(nlabels, device=device)
if config['z_dist']['type'] == 'gauss':
    zdist = get_zdist(dist_name=config['z_dist']['type'],dim=config['z_dist']['dim'], device=device)
elif config['z_dist']['type'] == 'multivariate_normal':
    zdist = get_zdist(dist_name=config['z_dist']['type'], dim=config['z_dist']['dim'], mean=mean, cov=cov, device=device)
elif config['z_dist']['type'] == 'gmm':
    zdist = get_zdist(dist_name=config['z_dist']['type'], 
                        dim=config['z_dist']['dim'], 
                        gmm_components_weight=gmm_components_weight, 
                        gmm_mean=gmm_mean, 
                        gmm_cov=gmm_cov, 
                        device=device)
else:
    raise NotImplementedError
print('noise type: ', config['z_dist']['type'])



# set eval mode 
generator.eval()
# visualize latentvecs
batch_size = config['training']['batch_size']
for index in tqdm(range(args.num_of_batches)):
    y = ydist.sample((batch_size,)).to(device)
    z = zdist.sample((batch_size,)).to(device)
    with torch.no_grad():
        gen_images = generator(z, y)
    gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
    torchvision.utils.save_image(gen_images, out_dir + '/batch_' + str(index) + '.png', nrow=8)
