import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_im2latent
from collections import OrderedDict
import torchvision
from tensorboardX import SummaryWriter
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
load_from = config['training']['load_from']
batch_size = config['training']['batch_size']
total_epochs = config['training']['total_epochs']
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
    label_path=config['data']['label_path'],
    size=config['data']['img_size'],
    simple_transform=config['data']['simple_transform']
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=False, pin_memory=True, sampler=None, drop_last=True
)

val_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['val_dir'],
    label_path=config['data']['label_path'],
    size=config['data']['img_size'],
    simple_transform=config['data']['simple_transform']
)
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=False, pin_memory=True, sampler=None, drop_last=True
)


# Number of labels
nlabels = config['data']['nlabels']


# Create models
img2vec = build_im2latent(config)
print(img2vec)

if load_from is not None:
    print('loading img2vec ckpt: ', load_from)
    loaded_dict = torch.load(load_from)
    img2vec.load_state_dict(loaded_dict)
    print('img2vec ckpt loaded!')


# Put models on gpu if needed
img2vec = img2vec.to(device)



# optimize and loss
optimizer = torch.optim.Adam(img2vec.parameters(), lr=lr, betas=(0.9, 0.999))
criterion = nn.MSELoss()


# Logger(tensorboard)
writer = SummaryWriter(logdir=out_dir+'/monitoring')





# Learning rate scheduler
milestones_step = config['training']['lr_scheduler']['milestones_step']
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(milestones_step,total_epochs,milestones_step), gamma=config['training']['lr_scheduler']['gamma'])




# Training loop
print('Start training...')
it = 0
# set mode
img2vec.train()
for epoch in range(total_epochs):
    print('Start epoch %d...' % epoch)

    for index ,(images, latentvecs) in enumerate(train_loader):
        it += 1
        # image -> vec
        pred_z = img2vec(images.to(device))

        # compute loss
        loss = criterion(pred_z, latentvecs.to(device))


        # img2vec model updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Print stats
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train loss', loss.cpu().data.numpy(), it)
        writer.add_scalar('lr', current_lr, it)
        print('[epoch %4d iter %4d] loss = %.4f'% (epoch, index, loss.cpu().data.numpy()))

    lr_scheduler.step()
    # Save checkpoint if necessary
    if (epoch+1) % save_per_epoch == 0 :
        print('Saving checkpoint...')
        torch.save(img2vec.state_dict(), out_dir + '/chkpts/epoch_'+ str(epoch+1) + '_im2vec.pth')
        # compute loss
        img2vec.eval()
        val_loss = []
        for images, latentvecs in val_loader:
            with torch.no_grad():
                pred_z = img2vec(images.to(device))
                loss = criterion(pred_z, latentvecs.to(device))

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)
        writer.add_scalar('val loss', val_loss, epoch)
        print('[epoch %4d] val loss = %.4f'% (epoch, val_loss))
        img2vec.train()


# save latentvecs
img2vec.eval()
for index, (x_real, y) in enumerate(train_loader):
    with torch.no_grad():
        z = img2vec(x_real.to(device))
    latentvecs = z.detach().cpu().numpy()
    np.save(out_dir + '/latentvecs/batch_'+ str(index+1)+'_latentvecs.npy', latentvecs)



        
    