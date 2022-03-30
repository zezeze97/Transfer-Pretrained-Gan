import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.logger import Logger
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_generator, build_im2latent
from gan_training.kl_loss import kl_divergence
from collections import OrderedDict
import torchvision
from tensorboardX import SummaryWriter
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from itertools import chain
import numpy as np


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
parser.add_argument('--generator_ckpt', type=str, help='Path to generator ckpt.')
parser.add_argument('--latentdir', type=str, help='Root path of latentvecs.')
parser.add_argument('--outdir', type=str, help='Save path.')
args = parser.parse_args()

config = load_config(args.config, default_path=None)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)


# Create missing directories
out_dir = args.outdir
latentvecs_root_dir = args.latentdir
if not path.exists(out_dir):
    os.makedirs(out_dir)

# use gpu
device = torch.device("cuda:0" if is_cuda else "cpu")


# Number of labels
nlabels = 1


# Create models
generator = build_generator(config)
print(generator)



# Put models on gpu if needed
generator = generator.to(device)


# Load generator ckpt
print('Loading pretrained generator...')
if config['training']['omit_embedding_layer']:
    print('omit embedding_layer of generator!')
    pretrained_ckpt = args.generator_ckpt
    loaded_dict = torch.load(pretrained_ckpt)
    pretrained_generator_state_dict = remove_module_str_in_state_dict(loaded_dict['generator'])
    generator_state_dict = generator.state_dict()
    new_dict = {k: v for k, v in pretrained_generator_state_dict.items() if k != 'embedding.weight'}
    generator_state_dict.update(new_dict)
    generator.load_state_dict(generator_state_dict)
else:
    pretrained_ckpt = args.generator_ckpt
    loaded_dict = torch.load(pretrained_ckpt)
    generator.load_state_dict(remove_module_str_in_state_dict(loaded_dict))
print('Pretrained generator loaded!')

# set eval mode 
generator.eval()

# visualize latentvecs
latentvecs_list = os.listdir(args.latentdir)
for index, filename in enumerate(latentvecs_list):
    latentvecs = np.load(path.join(args.latentdir,filename))
    # numpy -> torch
    latentvecs = torch.from_numpy(latentvecs).to(device)
    bs = latentvecs.size(0)
    y = torch.zeros(bs, dtype=torch.int64).to(device)

    with torch.no_grad():
        gen_images = generator(latentvecs, y)
    gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
    torchvision.utils.save_image(gen_images, out_dir + '/batch_' + str(index) + '.png', nrow=8)
