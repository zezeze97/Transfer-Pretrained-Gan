import argparse
import os
from os import path
import copy
import numpy as np
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist, interpolate_sphere
from gan_training.config import (
    load_config, build_models
)

# Arguments
parser = argparse.ArgumentParser(
    description='Create interpolations for a trained GAN.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
interp_dir = path.join(out_dir, 'test', 'interp')

# Creat missing directories
if not path.exists(interp_dir):
    os.makedirs(interp_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

# Get model file
model_file = config['test']['model_file']

# Models
device = torch.device("cuda:0" if is_cuda else "cpu")

generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Distributions
ydist = get_ydist(nlabels, device=device)
if config['z_dist']['type'] == 'gauss':
    zdist = get_zdist(dist_name=config['z_dist']['type'],dim=config['z_dist']['dim'], device=device)
elif config['z_dist']['type'] == 'multivariate_normal':
    mean_path = config['z_dist']['mean_path']
    cov_path = config['z_dist']['cov_path']
    mean = torch.FloatTensor(np.load(mean_path))
    cov = torch.FloatTensor(np.load(cov_path))
    zdist = get_zdist(dist_name=config['z_dist']['type'], dim=config['z_dist']['dim'], mean=mean, cov=cov, device=device)
elif config['z_dist']['type'] == 'gmm':
    gmm_components_weight = np.load(config['z_dist']['gmm_components_weight'])
    gmm_mean = np.load(config['z_dist']['gmm_mean'])
    gmm_cov = np.load(config['z_dist']['gmm_cov'])
    zdist = get_zdist(dist_name=config['z_dist']['type'], 
                        dim=config['z_dist']['dim'], 
                        gmm_components_weight=gmm_components_weight, 
                        gmm_mean=gmm_mean, 
                        gmm_cov=gmm_cov, 
                        device=device)
elif config['z_dist']['type'] == 'kde':
    # load latent vectors npy file
    latentvec_dir = config['z_dist']['latentvec_dir']
    for i,filename in enumerate(os.listdir(latentvec_dir)):
        if i == 0:
            latentvecs = np.load(os.path.join(latentvec_dir, filename))
        else:
            current_vecs = np.load(os.path.join(latentvec_dir, filename))
            latentvecs = np.concatenate((current_vecs,latentvecs),axis=0)

    print('latentvecs shape: ', latentvecs.shape)
    zdist = get_zdist(dist_name='kde', dim=config['z_dist']['dim'], latentvecs=latentvecs, device=device)

elif config['z_dist']['type'] == 'gmm2gauss':
    gmm_components_weight = np.load(config['z_dist']['gmm_components_weight'])
    gmm_mean = np.load(config['z_dist']['gmm_mean'])
    gmm_cov = np.load(config['z_dist']['gmm_cov'])
    zdist = get_zdist(dist_name=config['z_dist']['type'], 
                        dim=config['z_dist']['dim'], 
                        gmm_components_weight=gmm_components_weight, 
                        gmm_mean=gmm_mean, 
                        gmm_cov=gmm_cov, 
                        device=device)

else:
    raise NotImplementedError
print('noise type: ', config['z_dist']['type'])


# Load checkpoint if existant
load_dict = checkpoint_io.load(model_file)
it = load_dict.get('it', -1)
epoch_idx = load_dict.get('epoch_idx', -1)

# Interpolations
print('Creating interplations...')
nsteps = 3
nsubsteps = 10

y = ydist.sample((sample_size,))
zs = [zdist.sample((sample_size,)) for i in range(nsteps)]
ts = np.linspace(0, 1, nsubsteps)

it = 0
for z1, z2 in zip(zs, zs[1:] + [zs[0]]):
    for t in ts:
        # z = interpolate_sphere(z1, z2, float(t))
        z = t * z1 + (1-t) * z2
        with torch.no_grad():
            x = generator_test(z, y)
            utils.save_images(x, path.join(interp_dir, '%04d.png' % it),
                              nrow=sample_nrow)
            it += 1
            print('%d/%d done!' % (it, nsteps * nsubsteps))
