import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_generator, build_im2latent
from gan_training.distributions import get_ydist, get_zdist
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
total_iters = config['training']['total_iters']
lr = config['training']['lr']
out_dir = config['training']['out_dir']
save_per_iter = config['training']['save_per_iter']
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
        shuffle=False, pin_memory=True, sampler=None, drop_last=True
)


# Number of labels
nlabels = config['data']['nlabels']


# Create models
generator = build_generator(config)
img2vec = build_im2latent(config)
print(generator)
print(img2vec)

if load_from is not None:
    print('loading img2vec ckpt: ', load_from)
    loaded_dict = torch.load(load_from)
    img2vec.load_state_dict(loaded_dict)
    print('img2vec ckpt loaded!')


# Put models on gpu if needed
generator = generator.to(device)
img2vec = img2vec.to(device)



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




# Learning rate scheduler
step_size = config['training']['lr_scheduler']['step_size']
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=config['training']['lr_scheduler']['gamma'])

# z_dist, y_dist
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(dist_name=config['z_dist']['type'],dim=config['z_dist']['dim'], device=device)

# fix param in generator
for params in generator.parameters():
    params.requires_grad = False


# Training loop
print('Start training...')
# set mode
img2vec.train()
generator.eval()


# load label_embedding if needed

if config['training']['label_embedding'] is not None:
    print('loading label embedding...')
    label_embedding = torch.FloatTensor(np.load(config['training']['label_embedding'])).to(device)
    label_dim = config['generator']['kwargs']['embed_size']
    print('finish loading label_embedding!!')
for it in range(total_iters):
    print('Start iter %d...' % it)

    # sample y,z 
    sample_y = ydist.sample((batch_size,))
    sample_z = zdist.sample((batch_size,))

    # vec -> image
    with torch.no_grad():
        images = generator(sample_z, sample_y)
    images = images.detach()

    # image -> vec
    pred_z = img2vec(images)

    # compute loss
    loss = criterion(pred_z, sample_z)


    # img2vec model updates
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print stats
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('loss', loss.cpu().data.numpy(), it)
    writer.add_scalar('lr', current_lr, it)
    print('[it %4d] loss = %.4f'% (it, loss.cpu().data.numpy()))

    lr_scheduler.step()
    # Save checkpoint if necessary
    if (it+1) % save_per_iter == 0 :
        print('Saving checkpoint...')
        torch.save(img2vec.state_dict(), out_dir + '/chkpts/iter_'+ str(it+1) + '_im2vec.pth')
        #===visualize current result====
        # read a batch of data
        x_real, y = next(iter(train_loader))
        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # draw gt images
        gt_images = torchvision.utils.make_grid(x_real*0.5+0.5, nrow=8, padding=2)
        writer.add_image('gt_images', gt_images, global_step = it+1)

        # gt_image -> latentvecs -> generated image
        img2vec.eval()
        with torch.no_grad():
            z = img2vec(x_real)
            if nlabels > 1:
                temp_bs = z.size(0)
                
                y = torch.zeros((temp_bs, label_dim)).to(device)
                for i in range(temp_bs):
                    y[i,:] = label_embedding

            gen_images = generator(z, y)

        # draw generated images
        gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
        writer.add_image('gen_images', gen_images, global_step = it+1)
        img2vec.train()


# save latentvecs
img2vec.eval()
for index, (x_real, y) in enumerate(train_loader):
    with torch.no_grad():
        z = img2vec(x_real)
    latentvecs = z.detach().cpu().numpy()
    np.save(out_dir + '/latentvecs/batch_'+ str(index+1)+'_latentvecs.npy', latentvecs)



        
    