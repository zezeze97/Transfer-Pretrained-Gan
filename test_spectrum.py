import argparse
import os
from os import path
from collections import OrderedDict
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)
import numpy as np
import random


SEED=999
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(SEED)

def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename

# Arguments
parser = argparse.ArgumentParser(
    description='Test a trained GAN and compute spectrum.'
)
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--num_sample', type=int, help='Num of sample to compute spectrum.')
parser.add_argument('--patience', type=float)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
# force to extract feature in discriminator
config['discriminator']['kwargs']['extract_feature'] = True

is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
sample_size = config['test']['sample_size']


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

# Load checkpoint
load_dict = torch.load(os.path.join(out_dir, 'chkpts', model_file))
pretrained_generator_loaded_dict = remove_module_str_in_state_dict(load_dict['generator'])
generator.load_state_dict(pretrained_generator_loaded_dict)
print('pretrained generator loaded!')
pretrained_discriminator_loaded_dict = remove_module_str_in_state_dict(load_dict['discriminator'])
discriminator.load_state_dict(pretrained_discriminator_loaded_dict)
print('pretrained discriminator loaded!')

features = []
with torch.no_grad():
    for i in range(args.num_sample):
        z = zdist.sample((1,))
        y = ydist.sample((1,))
        fake_img = generator(z, y)
        logits, feature = discriminator(fake_img, y)
        features.append(feature)
features = torch.concat(features, dim=0)
u,s,v = torch.svd(features)
total_index = s.shape[0]
theSum = torch.sum(s)
for i in range(total_index, 0, -1):
    # print(i, torch.sum(s[:i]) / theSum)
    if (torch.sum(s[:i]) / theSum) < 1 - args.patience:
        print(i+1)
        break
    
    

