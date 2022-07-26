import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)
import numpy as np

# Arguments
parser = argparse.ArgumentParser(
    description='Test a trained GAN and create visualizations.'
)
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--test_imagenet', action='store_true', help='Test imagenet.')
parser.add_argument('--pretrained_ckpt_path', type=str, help='Path to pretrained ckpt if need to test imagenet.', default=None)
parser.add_argument('--imagenet_path', type=str, help='Path to imagenet path', default=None)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')


args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
if args.test_imagenet:
    nlabels = 1000
    config['data']['nlabels'] = 1000
    config['z_dist']['type'] = 'gauss'
    config['test']['conditional_samples'] = True
else:
    nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
img_dir = path.join(out_dir, 'test', 'img')
img_all_dir = path.join(out_dir, 'test', 'img_all')
fid_test_dir = path.join(out_dir, 'test', 'fid_fake_imgs')
if args.test_imagenet:
    fid_fake_imgs_num = len(os.listdir(args.imagenet_path))

else:
    fid_fake_imgs_num = config['test']['fid_fake_imgs_num']
# Creat missing directories
if not path.exists(img_dir):
    os.makedirs(img_dir)
if not path.exists(img_all_dir):
    os.makedirs(img_all_dir)

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

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# load mean and cov of latentvecs if neccessary
if config['z_dist']['type'] == 'multivariate_normal':
    mean_path = config['z_dist']['mean_path']
    cov_path = config['z_dist']['cov_path']
    mean = torch.FloatTensor(np.load(mean_path))
    cov = torch.FloatTensor(np.load(cov_path))
# load gmm parameters if neccessary
if config['z_dist']['type'] == 'gmm':
    gmm_components_weight = np.load(config['z_dist']['gmm_components_weight'])
    gmm_mean = np.load(config['z_dist']['gmm_mean'])
    gmm_cov = np.load(config['z_dist']['gmm_cov'])


# Distributions
ydist = get_ydist(nlabels, device=device)
if config['z_dist']['type'] == 'gauss':
    zdist = get_zdist(dist_name=config['z_dist']['type'],dim=config['z_dist']['dim'], device=device)
elif config['z_dist']['type'] == 'multivariate_normal':
    mean_path = config['z_dist']['mean_path']
    cov_path = config['z_dist']['cov_path']
    mean = torch.FloatTensor(np.load(mean_path))
    cov = torch.FloatTensor(np.load(cov_path))
    zdist = get_zdist(dist_name=config['z_dist']['type'], dim=config['z_dist']['dim'], mean=mean, cov=cov, device=device)
elif config['z_dist']['type'] == 'gmm':
    gmm_components_weight = np.load(config['z_dist']['gmm_components_weight'])
    gmm_mean = np.load(config['z_dist']['gmm_mean'])
    gmm_cov = np.load(config['z_dist']['gmm_cov'])
    zdist = get_zdist(dist_name=config['z_dist']['type'], 
                        dim=config['z_dist']['dim'], 
                        gmm_components_weight=gmm_components_weight, 
                        gmm_mean=gmm_mean, 
                        gmm_cov=gmm_cov, 
                        device=device)
elif config['z_dist']['type'] == 'kde':
    # load latent vectors npy file
    latentvec_dir = config['z_dist']['latentvec_dir']
    for i,filename in enumerate(os.listdir(latentvec_dir)):
        if i == 0:
            latentvecs = np.load(os.path.join(latentvec_dir, filename))
        else:
            current_vecs = np.load(os.path.join(latentvec_dir, filename))
            latentvecs = np.concatenate((current_vecs,latentvecs),axis=0)

    print('latentvecs shape: ', latentvecs.shape)
    zdist = get_zdist(dist_name='kde', dim=config['z_dist']['dim'], latentvecs=latentvecs, device=device)

elif config['z_dist']['type'] == 'gmm2gauss':
    gmm_components_weight = np.load(config['z_dist']['gmm_components_weight'])
    gmm_mean = np.load(config['z_dist']['gmm_mean'])
    gmm_cov = np.load(config['z_dist']['gmm_cov'])
    zdist = get_zdist(dist_name=config['z_dist']['type'], 
                        dim=config['z_dist']['dim'], 
                        gmm_components_weight=gmm_components_weight, 
                        gmm_mean=gmm_mean, 
                        gmm_cov=gmm_cov, 
                        device=device)

else:
    raise NotImplementedError
print('noise type: ', config['z_dist']['type'])


# Evaluator
zdist_type=config['z_dist']['type']
evaluator = Evaluator(generator_test, zdist_type, zdist, ydist,
                      batch_size=batch_size, device=device)

# Load checkpoint if existant

# change class embedding layer if test imagenet
if args.test_imagenet:
    pretrained_class_embedding = torch.load(args.pretrained_ckpt_path)['generator']['module.embedding.weight']
    finetune_stat_dict = torch.load(os.path.join(config['training']['out_dir'], "chkpts", config['test']['model_file']))['generator']
    finetune_stat_dict['module.embedding.weight'] = pretrained_class_embedding
    generator_test.load_state_dict(finetune_stat_dict)
    it = -1
else:
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

# Inception score
if config['test']['compute_inception']:
    print('Computing inception score...')
    inception_mean, inception_std = evaluator.compute_inception_score()
    print('Inception score: %.4f +- %.4f' % (inception_mean, inception_std))

# FID
if config['test']['compute_fid']:
    # generate and save fake images
    print('Generating fake images to compute fid...')
    evaluator.save_samples(sample_num=fid_fake_imgs_num, save_dir=fid_test_dir)
    print('Computing FID score...')
    fid_img_size = (config['data']['img_size'], config['data']['img_size'])

    if args.test_imagenet:
        gt_path = args.imagenet_path
    else:
        gt_path = config['data']['test_dir'] + '/0/'
        
        

    fid = evaluator.compute_fid_score(generated_img_path = fid_test_dir, 
                                        gt_path = gt_path, 
                                        img_size = fid_img_size)
    print('FID: ', fid)

# Samples
print('Creating samples...')
ztest = zdist.sample((sample_size,))
x = evaluator.create_samples(ztest)
utils.save_images(x, path.join(img_all_dir, '%08d.png' % it),
                  nrow=sample_nrow)
if config['test']['conditional_samples']:
    for y_inst in tqdm(range(nlabels)):
        x = evaluator.create_samples(ztest, y_inst)
        utils.save_images(x, path.join(img_dir, '%04d.png' % y_inst),
                          nrow=sample_nrow)
