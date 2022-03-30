import argparse
import os
from os import path
import numpy as np
from collections import OrderedDict
import torch
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_im2latent


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--img2vec_model_ckpt', type=str, help='Path to img2vec_model_ckpt.')
parser.add_argument('--output_dir', type=str, help='Path to save mean and cov.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()

# short hands
config = load_config(args.config, default_path=None)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
out_dir = args.output_dir
ckpt_path = args.img2vec_model_ckpt
batch_size = config['training']['batch_size']
# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)

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

# build model
img2vec = build_im2latent(config)
# Put models on gpu if needed
img2vec = img2vec.to(device)
# load ckpt
img2vec.load_state_dict(torch.load(ckpt_path))

# feed all data to get latentvecs
img2vec.eval()
latentvecs_list = []
for x_real, y in train_loader:
    x_real, y = x_real.to(device), y.to(device)
    y.clamp_(None, nlabels-1)
    with torch.no_grad():
    # Img -> Latent
        z = img2vec(x_real)
    latentvecs_list.append(z.cpu().numpy())
# concat_result
for i,z in enumerate(latentvecs_list):
    np.save(out_dir + '/latentvecs/batch_'+str(i)+'.npy', z)
    if i == 0:
        latentvecs = z
    else:
        current_vecs = z
        latentvecs = np.concatenate((latentvecs, current_vecs), axis=0)
# compute mean and cov of latentvecs and save
mean = np.mean(latentvecs, axis= 0)
cov = np.cov(latentvecs, rowvar=False)
print('mean shape: ', mean.shape)
print('cov shape: ', cov.shape)
np.save(out_dir + '/mean.npy', mean)
np.save(out_dir + '/cov.npy', cov)

