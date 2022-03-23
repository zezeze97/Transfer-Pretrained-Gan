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
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, default_path=None)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Short hands
batch_size = config['training']['batch_size']
iter_per_batch = config['training']['iter_per_batch']
lr = config['training']['lr']
out_dir = config['training']['out_dir']
checkpoint_dir = path.join(out_dir, 'chkpts')
latentvecs_dir = path.join(out_dir,'latentvecs')
use_regularization = config['training']['use_regularization']
regularization_lambda = config['training']['regularization']['lambda']
# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not path.exists(latentvecs_dir):
    os.makedirs(latentvecs_dir)

# use gpu
device = torch.device("cuda:0" if is_cuda else "cpu")


# Dataset
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    lsun_categories=config['data']['lsun_categories_train'],
    simple_transform=config['data']['simple_transform']
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=False, pin_memory=True, sampler=None, drop_last=True
)

# Number of labels
nlabels = min(nlabels, config['data']['nlabels'])


# Create models
generator = build_generator(config)
# embedding layer to find latentvecs
latent_vecs_embedding_layer = nn.Embedding(len(train_dataset), config['z_dist']['dim'])
print(generator)


# Put models on gpu if needed
generator = generator.to(device)
latent_vecs_embedding_layer = latent_vecs_embedding_layer.to(device)


# Logger(tensorboard)
writer = SummaryWriter(logdir=out_dir+'/monitoring')


# Load generator ckpt
pretrained_ckpt = config['training']['pretrain_ckpt_file']
loaded_dict = torch.load(pretrained_ckpt)
print('Loading pretrained generator...')
generator.load_state_dict(remove_module_str_in_state_dict(loaded_dict['generator']))
print('Pretrained generator loaded!')
it = -1


# fix param in generator
for params in generator.parameters():
    params.requires_grad = False


# Training loop
print('Start training...')
for index, (x_real, y) in enumerate(train_loader):
    # optimize and loss
    optimizer = torch.optim.Adam(latent_vecs_embedding_layer.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_scheduler']['step_size'], gamma=config['training']['lr_scheduler']['gamma'])
    # set mode
    latent_vecs_embedding_layer.train()
    generator.eval()
    print('Start Batch %d...' % (index+1))
    for iteration in range(iter_per_batch):
        it += 1
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rates', current_lr, it)
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # get current batch latentvecs
        latent_list = list(range(index*batch_size, min((index+1)*batch_size, len(train_dataset))))
        latent_list = torch.tensor(latent_list).to(device)
        z = latent_vecs_embedding_layer(latent_list)

        # compute current batch mean and cov if needed
        if use_regularization:
            z_mean = torch.mean(z, dim=0)
            z_cov = torch.cov(z.T)
        
        # Latent -> Img
        x_generate = generator(z, y)

        # compute loss
        image_loss = criterion(x_generate, x_real)
        if use_regularization:
            if config['training']['regularization']['type'] == 'kl':
                regularization_loss = kl_divergence(z_mean, z_cov, config['z_dist']['dim'], eps = 0.0000000001)
            if config['training']['regularization']['type'] == 'l2':
                standard_cov = torch.eye(config['z_dist']['dim']).to(device)
                regularization_loss = torch.linalg.norm(z_mean) + torch.linalg.norm(z_cov - standard_cov)
            total_loss = image_loss + regularization_lambda * regularization_loss
        else:
            total_loss = image_loss

        # img2vec model updates
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        writer.add_scalar('total loss', total_loss.cpu().data.numpy(), it)

        # Print stats
        if use_regularization:
            writer.add_scalar('total loss', total_loss.cpu().data.numpy(), it)
            writer.add_scalar('image loss', image_loss.cpu().data.numpy(), it)
            writer.add_scalar('regularization loss', regularization_loss.cpu().data.numpy(), it)
            print('[batch %0d, it %4d] total loss = %.4f image loss = %.4f regularization loss = %.4f'% (index+1, iteration+1, total_loss.cpu().data.numpy(), image_loss.cpu().data.numpy(), regularization_loss.cpu().data.numpy()))
        else:
            writer.add_scalar('total loss', total_loss.cpu().data.numpy(), it)
            writer.add_scalar('image loss', image_loss.cpu().data.numpy(), it)
            print('[batch %0d, it %4d] total loss = %.4f image loss = %.4f'% (index+1, iteration+1, total_loss.cpu().data.numpy(), image_loss.cpu().data.numpy()))
        lr_scheduler.step()


    # Save current batch latentvecs and visualize
    latent_vecs_embedding_layer.eval()
    with torch.no_grad():
        # get current batch latentvecs and generated fake image
        latent_list = list(range(index*batch_size, min((index+1)*batch_size, len(train_dataset))))
        latent_list = torch.tensor(latent_list).to(device)
        z = latent_vecs_embedding_layer(latent_list)
        gen_images = generator(z, y)
    latentvecs = z.detach().cpu().numpy()
    np.save(out_dir + '/latentvecs/batch_'+ str(index+1)+'_latentvecs.npy', latentvecs)
    # draw gt images
    gt_images = torchvision.utils.make_grid(x_real*0.5+0.5, nrow=8, padding=2)
    writer.add_image('gt_images', gt_images, global_step = index+1)
    gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
    writer.add_image('gen_images', gen_images, global_step = index+1)


        




        
    