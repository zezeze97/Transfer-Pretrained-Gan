import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_generator
from collections import OrderedDict
import torchvision
from tensorboardX import SummaryWriter
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
epochs = config['training']['epochs']
lr = config['training']['lr']
out_dir = config['training']['out_dir']
save_per_epoch = config['training']['save_per_epoch']
checkpoint_dir = path.join(out_dir, 'chkpts')
latentvecs_dir = path.join(out_dir,'latentvecs')
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
        shuffle=False, pin_memory=True, sampler=None, drop_last=False
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


# fix param in generator
if config['training']['fix_class_embedding']:
    for params in generator.parameters():
        params.requires_grad = False
else:
    # set class embedding layer learnable in generator
    for k,v in generator.named_parameters():
        if k =='embedding.weight':
            v.requires_grad = True
        else:
            v.requires_grad = False
        




# optimizer and loss
if config['training']['fix_class_embedding']:
    optimizer = torch.optim.Adam(latent_vecs_embedding_layer.parameters(), lr=lr, betas=(0.9, 0.999))
else:
    optimizer = torch.optim.Adam(chain(latent_vecs_embedding_layer.parameters(),filter(lambda p: p.requires_grad, generator.parameters())), lr=lr, betas=(0.9, 0.999))
criterion = nn.MSELoss()


# Logger(tensorboard)
writer = SummaryWriter(logdir=out_dir+'/monitoring')


# Load generator ckpt
print('Loading pretrained generator...')
print('Change class embedding layer of generator!')
pretrained_ckpt = config['training']['pretrain_ckpt_file']
pretrained_generator_state_dict = torch.load(pretrained_ckpt)
generator_state_dict = generator.state_dict()
new_dict = {k: v for k, v in pretrained_generator_state_dict.items() if k != 'embedding.weight'}

# Special class embedding init
if config['training']['class_embedding'] is not None:
    print('using special class embedding init!')
    class_embedding = torch.FloatTensor(np.load(config['training']['class_embedding']))
    # (256,) -> (1,256)
    class_embedding = torch.unsqueeze(class_embedding, dim=0)
    new_dict['embedding.weight'] = class_embedding 
generator_state_dict.update(new_dict)
generator.load_state_dict(generator_state_dict)
print('Pretrained generator loaded!')
it = -1



# Learning rate scheduler
milestones_step = config['training']['lr_scheduler']['milestones_step']
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(milestones_step,epochs,milestones_step), gamma=config['training']['lr_scheduler']['gamma'])




# Training loop
print('Start training...')

for epoch in range(epochs):
    print('Start epoch %d...' % epoch)

    # set mode
    latent_vecs_embedding_layer.train()
    generator.eval()
    if not config['training']['fix_class_embedding']:
        # Setting generator class embedding layer trainable
        generator.embedding.train()
        
    

    for index, (x_real, y) in enumerate(train_loader):
        it += 1
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rates', current_lr, it)
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # get current batch latentvecs
        latent_list = list(range(index*batch_size, min((index+1)*batch_size, len(train_dataset))))
        latent_list = torch.tensor(latent_list).to(device)
        z = latent_vecs_embedding_layer(latent_list)
        # Latent -> Img
        x_generate = generator(z, y)

        # compute loss
        image_loss = criterion(x_generate, x_real)

        # img2vec model updates
        optimizer.zero_grad()
        image_loss.backward()
        optimizer.step()
        
        writer.add_scalar('image loss', image_loss.cpu().data.numpy(), it)
        print('[epoch %0d, it %4d] image loss = %.4f'% (epoch, it, image_loss.cpu().data.numpy()))
    lr_scheduler.step()
    # Save checkpoint if necessary
    if (epoch+1) % save_per_epoch == 0 :
        print('Saving checkpoint...')
        torch.save(latent_vecs_embedding_layer.state_dict(), out_dir + '/chkpts/latent_vecs_embedding_layer.pt')
        torch.save(generator.embedding.state_dict(), out_dir + '/chkpts/generator_class_embedding.pt')
        #===visualize current result====
        # read a batch of data
        x_real, y = next(iter(train_loader))
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # draw gt images
        gt_images = torchvision.utils.make_grid(x_real*0.5+0.5, nrow=8, padding=2)
        writer.add_image('gt_images', gt_images, global_step = epoch+1)

        # latentvecs -> generated image
        latent_vecs_embedding_layer.eval()
        generator.eval()
        with torch.no_grad():
            latent_list = list(range(batch_size))
            latent_list = torch.tensor(latent_list).to(device)
            z = latent_vecs_embedding_layer(latent_list)
            gen_images = generator(z, y)

        # draw generated images
        gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
        writer.add_image('gen_images', gen_images, global_step = epoch+1)


# save latentvecs
latent_vecs_embedding_layer.eval()
with torch.no_grad():
    # get current batch latentvecs
    latent_list = list(range(len(train_dataset)))
    latent_list = torch.tensor(latent_list).to(device)
    z = latent_vecs_embedding_layer(latent_list)
latentvecs = z.detach().cpu().numpy()
print('Saved latentvecs shape: ', latentvecs.shape)
np.save(out_dir + '/latentvecs/latentvecs.npy', latentvecs)