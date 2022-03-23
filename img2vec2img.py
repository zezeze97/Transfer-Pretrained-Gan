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
load_from = config['training']['load_from']
use_pretrained_img_feature_extractor = config['training']['use_pretrained_img_feature_extractor']
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
lr = config['training']['lr']
out_dir = config['training']['out_dir']
save_per_epoch = config['training']['save_per_epoch']
checkpoint_dir = path.join(out_dir, 'chkpts')
use_regularization = config['training']['use_regularization']
reularization_lambda = config['training']['regularization']['lambda']
# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

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
    image_feature_extractor = models.resnet18(pretrained=True)
    return_nodes = {'layer4.1.relu_1': 'layer4',
                'layer3.1.relu_1': 'layer3',
                'layer2.1.relu_1': 'layer2',
                'layer1.1.relu_1': 'layer1' }
    image_feature_extractor = create_feature_extractor(image_feature_extractor, return_nodes=return_nodes)
if load_from is not None:
    print('loading img2vec ckpt: ', load_from)
    loaded_dict = torch.load(load_from)
    img2vec.load_state_dict(loaded_dict)
    print('img2vec ckpt loaded!')


# Put models on gpu if needed
generator = generator.to(device)
img2vec = img2vec.to(device)
if use_pretrained_img_feature_extractor:
    image_feature_extractor = image_feature_extractor.to(device)

# optimize and loss
optimizer = torch.optim.Adam(img2vec.parameters(), lr=lr, betas=(0.9, 0.999))
criterion = nn.MSELoss()


# Logger(tensorboard)
writer = SummaryWriter(logdir=out_dir+'/monitoring')


# Load generator ckpt
pretrained_ckpt = config['training']['pretrain_ckpt_file']
loaded_dict = torch.load(pretrained_ckpt)
print('Loading pretrained generator...')
generator.load_state_dict(remove_module_str_in_state_dict(loaded_dict['generator']))
print('Pretrained generator loaded!')
it = -1



# Learning rate scheduler
milestones_step = config['training']['lr_scheduler']['milestones_step']
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(milestones_step,epochs,milestones_step), gamma=config['training']['lr_scheduler']['gamma'])


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
for epoch in range(epochs):
    print('Start epoch %d...' % epoch)

    for x_real, y in train_loader:
        it += 1
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rates', current_lr, it)
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # Img -> Latent
        z = img2vec(x_real)

        # compute mean and cov of cunrrent batch latentvecs if needed
        if use_regularization:
            z_mean = torch.mean(z, dim=0)
            z_cov = torch.cov(z.T)
        
        # Latent -> Img
        x_generate = generator(z, y)

        # compute loss
        if use_pretrained_img_feature_extractor:
            # compute senmantic output of x_real and x_generate
            with torch.no_grad():
                senmantic_real = image_feature_extractor(x_real)
            senmantic_gen = image_feature_extractor(x_generate)
            image_loss = 0
            for k in return_nodes.keys():
                image_loss += criterion(senmantic_gen[return_nodes[k]], senmantic_real[return_nodes[k]].detach())
        else:
            image_loss = criterion(x_generate, x_real)

        if use_regularization:
            if config['training']['regularization']['type'] == 'kl':
                regularization_loss = kl_divergence(z_mean, z_cov, config['z_dist']['dim'], eps = 0.0000000001)
            if config['training']['regularization']['type'] == 'l2':
                standard_cov = torch.eye(config['z_dist']['dim']).to(device)
                regularization_loss = torch.linalg.norm(z_mean) + torch.linalg.norm(z_cov - standard_cov)
            total_loss = image_loss + reularization_lambda * regularization_loss
        else:
            total_loss = image_loss

        # img2vec model updates
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print stats
        if use_regularization:
            writer.add_scalar('total loss', total_loss.cpu().data.numpy(), it)
            writer.add_scalar('image loss', image_loss.cpu().data.numpy(), it)
            writer.add_scalar('regularization loss', regularization_loss.cpu().data.numpy(), it)
            print('[epoch %0d, it %4d] image loss = %.4f regularization loss =  %.4f'% (epoch, it, image_loss.cpu().data.numpy(), regularization_loss.cpu().data.numpy()))        
        else:
            writer.add_scalar('total loss', total_loss.cpu().data.numpy(), it)
            writer.add_scalar('image loss', image_loss.cpu().data.numpy(), it)
            print('[epoch %0d, it %4d] image loss = %.4f'% (epoch, it, image_loss.cpu().data.numpy()))

    lr_scheduler.step()
    # Save checkpoint if necessary
    if (epoch+1) % save_per_epoch == 0 :
        print('Saving checkpoint...')
        torch.save(img2vec.state_dict(), out_dir + '/chkpts/epoch_'+ str(epoch+1) + '_im2vec.pth')
        #===visualize current result====
        # read a batch of data
        x_real, y = next(iter(train_loader))
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # draw gt images
        gt_images = torchvision.utils.make_grid(x_real*0.5+0.5, nrow=8, padding=2)
        writer.add_image('gt_images', gt_images, global_step = epoch+1)

        # gt_image -> latentvecs -> generated image
        img2vec.eval()
        with torch.no_grad():
            z = img2vec(x_real)
            gen_images = generator(z, y)

        # draw generated images
        gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
        writer.add_image('gen_images', gen_images, global_step = epoch+1)
        img2vec.train()




        
    