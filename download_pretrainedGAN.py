import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
import scipy.io as sio
import numpy as np
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)

DATA = 'ImageNet'

if DATA == 'CELEBA':
    config_path = './configs/pretrained/celebA_pretrained.yaml'
elif DATA == 'ImageNet':
    config_path = './configs/pretrained/imagenet_pretrained.yaml'
elif DATA == 'Church':
    config_path = './configs/pretrained/lsun_church_pretrained.yaml'
elif DATA == 'Bridge':
    config_path = './configs/pretrained/lsun_bridge_pretrained.yaml'
elif DATA == 'Bedroom':
    config_path = './configs/pretrained/lsun_bedroom_pretrained.yaml'
elif DATA == 'Tower':
    config_path = './configs/pretrained/lsun_tower_pretrained.yaml'

save_dir = './pretrained_ckpt/'

no_cuda = 0

config = load_config(config_path, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not no_cuda)

# Creat missing dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Shorthands
out_dir = config['training']['out_dir']
checkpoint_dir = path.join(out_dir, 'chkpts')


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

print('Random init: ',generator.module.resnet_0_0.conv_0.weight[1,1,:,:])

# Load checkpoint if existant
load_dict = checkpoint_io.load(model_file)
it = load_dict.get('it', -1)
epoch_idx = load_dict.get('epoch_idx', -1)

print('Pretrained: ',generator.module.resnet_0_0.conv_0.weight[1,1,:,:])


if DATA == 'ImageNet':

    TrainModeSave = DATA
    torch.save(generator.module.state_dict(), save_dir + TrainModeSave + 'Pre_generator.pt')
    torch.save(discriminator.module.state_dict(), save_dir + TrainModeSave + 'Pre_discriminator.pt')








