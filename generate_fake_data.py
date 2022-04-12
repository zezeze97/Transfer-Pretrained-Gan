import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.config import load_config, build_generator
from gan_training.distributions import get_zdist, get_ydist
from collections import OrderedDict
import torchvision
import numpy as np
from tqdm import tqdm
import json

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
train_image_dir = path.join(out_dir, "image", "train")
val_image_dir = path.join(out_dir, "image", "val")
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(train_image_dir):
    os.makedirs(train_image_dir)
if not path.exists(val_image_dir):
    os.makedirs(val_image_dir)



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
pretrained_generator_loaded_dict = remove_module_str_in_state_dict(torch.load(config['training']['pretrain_ckpt_file'])['generator']) 
generator.load_state_dict(pretrained_generator_loaded_dict)
print('pretrained generator loaded!')




# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(dist_name=config['z_dist']['type'],dim=config['z_dist']['dim'], device=device)





# set eval mode 
generator.eval()
latentvecs_dict = {}
batch_size = config['training']['batch_size']
total_batches = args.num_of_batches
train_batches = int(total_batches*0.8)
val_batches = total_batches - train_batches

# generate train images
for index in tqdm(range(train_batches)):
    y = ydist.sample((batch_size,)).to(device)
    z = zdist.sample((batch_size,)).to(device)
    with torch.no_grad():
        gen_images = generator(z, y)
        gen_images = gen_images*0.5+0.5


    z = z.detach().cpu().numpy()
    for i in range(batch_size):
        torchvision.utils.save_image(gen_images[i,:,:,:], out_dir + '/image/train/train_fake_' + str(index*batch_size + i) + '.png')
        latentvecs_dict['train_fake_' + str(index*batch_size + i) + '.png'] = z[i,:]


# generate val images
for index in tqdm(range(val_batches)):
    y = ydist.sample((batch_size,)).to(device)
    z = zdist.sample((batch_size,)).to(device)
    with torch.no_grad():
        gen_images = generator(z, y)
        gen_images = gen_images*0.5+0.5


    z = z.detach().cpu().numpy()
    for i in range(batch_size):
        torchvision.utils.save_image(gen_images[i,:,:,:], out_dir + '/image/val/val_fake_' + str(index*batch_size + i) + '.png')
        latentvecs_dict['val_fake_' + str(index*batch_size + i) + '.png'] = z[i,:]

np.save(path.join(out_dir,'latentvecs.npy'), latentvecs_dict)