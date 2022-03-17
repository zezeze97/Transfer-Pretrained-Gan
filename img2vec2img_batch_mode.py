import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.logger import Logger
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_generator, build_im2latent
from collections import OrderedDict
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from itertools import chain


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
use_pretrained_img_feature_extractor = config['training']['use_pretrained_img_feature_extractor']


# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(out_dir + '/latentvecs'):
    os.mkdir(out_dir + '/latentvecs')


# use gpu
device = torch.device("cuda:0" if is_cuda else "cpu")


# Dataset
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    lsun_categories=config['data']['lsun_categories_train']
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)

# Number of labels
nlabels = min(nlabels, config['data']['nlabels'])


# Create models
generator = build_generator(config)
img2vec = build_im2latent(config)
print(generator)
print(img2vec)
if use_pretrained_img_feature_extractor:
    image_feature_extractor = models.vgg16_bn(pretrained=True)
    return_nodes = {'flatten': 'flatten'}
    image_feature_extractor = create_feature_extractor(image_feature_extractor, return_nodes=return_nodes)

# Put models on gpu if needed
generator = generator.to(device)
img2vec = img2vec.to(device)
if use_pretrained_img_feature_extractor:
    image_feature_extractor = image_feature_extractor.to(device)




# Logger(tensorboard)
writer = SummaryWriter(logdir=out_dir+'/monitoring')


# Load generator ckpt
pretrained_ckpt = config['training']['pretrain_ckpt_file']
loaded_dict = torch.load(pretrained_ckpt)
print('Loading pretrained generator...')
generator.load_state_dict(remove_module_str_in_state_dict(loaded_dict['generator']))
print('Pretrained generator loaded!')
it = -1


if use_pretrained_img_feature_extractor:
    # fix param in generator and image_feature_extractor
    for params in chain(generator.parameters(),image_feature_extractor.parameters()):
        params.requires_grad = False
else:
    # fix param in generator
    for params in generator.parameters():
        params.requires_grad = False


# Training loop
print('Start training...')
# set mode
img2vec.train()
generator.eval()
if use_pretrained_img_feature_extractor:
    image_feature_extractor.eval()


for batch_index, (x_real, y) in enumerate(train_loader):
    # optimize and loss
    optimizer = torch.optim.Adam(img2vec.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_scheduler']['step_size'], gamma=config['training']['lr_scheduler']['gamma'])
    for iter in range(iter_per_batch):
        it += 1
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rates', current_lr, it)
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # Img -> Latent
        z = img2vec(x_real)
        
        # Latent -> Img
        x_generate = generator(z, y)
        
        if use_pretrained_img_feature_extractor:
            # compute senmantic output of x_real and x_generate
            with torch.no_grad():
                senmantic_real = image_feature_extractor(x_real)['flatten']
            senmantic_gen = image_feature_extractor(x_generate)['flatten']
            loss = criterion(senmantic_gen, senmantic_real)
        else:
            loss = criterion(x_generate, x_real)
        # img2vec model updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        writer.add_scalar('losses', loss.cpu().data.numpy(), it)

        # Print stats
        print('[batch %0d, iter %4d] loss = %.4f'% (batch_index+1, iter, loss.cpu().data.numpy()))

    # Save per batch result
    print('Saving current batch latentvecs and visualize...')
    # draw gt images
    gt_images = torchvision.utils.make_grid(x_real*0.5+0.5, nrow=8, padding=2)
    writer.add_image('gt_images', gt_images, global_step = batch_index+1)
    # gt_image -> latentvecs -> generated image
    img2vec.eval()
    with torch.no_grad():
        z = img2vec(x_real)
        gen_images = generator(z, y)
    # draw generated images
    gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
    writer.add_image('gen_images', gen_images, global_step = batch_index+1)
    # saving current batch latentvecs
    np.save(out_dir+'/latentvecs/'+'batch_'+str(batch_index+1)+'.npy',z.detach().cpu().numpy())
    img2vec.train()




        
    