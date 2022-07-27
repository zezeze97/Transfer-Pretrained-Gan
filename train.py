import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)
import numpy as np
import math
from collections import OrderedDict

def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename

def add_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = "module." + k # add module
        state_dict_rename[name] = v
    return state_dict_rename

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

if config['training']['pretrain_ckpt_file'] is not None:
    tmp_ckpt_file = config['training']['pretrain_ckpt_file'] 
    config['training']['pretrain_ckpt_file'] = os.path.join(
            os.path.split(os.path.abspath(__file__))[0], tmp_ckpt_file) 
    print(config['training']['pretrain_ckpt_file']) 

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
fid_every = config['training']['fid_every']
fid_fake_imgs_num = config['training']['fid_fake_imgs_num']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_nlabels = config['training']['sample_nlabels']
out_dir = config['training']['out_dir']
checkpoint_dir = path.join(out_dir, 'chkpts')
change_generator_embedding_layer = config['training']['change_generator_embedding_layer']
change_discriminator_fc_layer = config['training']['change_discriminator_fc_layer']
max_epoch = config['training']['max_epoch']
max_iter = config['training']['max_iter']
# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

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
sample_nlabels = min(nlabels, sample_nlabels)

# Create models
generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, config
)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
)

# Get model file
model_file = config['training']['model_file']

# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)


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

# Save for tests
ntest = batch_size
x_real, ytest = utils.get_nsamples(train_loader, ntest)
ytest.clamp_(None, nlabels-1)
if config['z_dist']['type'] == 'gmm2gauss':
    ztest = zdist.sample((ntest,), use_gmm=True)
else:
    ztest = zdist.sample((ntest,))
utils.save_images(x_real, path.join(out_dir, 'real.png'))



# Train
tstart = t0 = time.time()

# Load pretrained ckpt
finetune_mode = config['training']['finetune']
if finetune_mode:
    if change_generator_embedding_layer and change_discriminator_fc_layer:
        print('change generator embedding layer and discriminator fc layer!!!')
        if config['training']['pretrain_ckpt_file'] is None:
            # load pretrained generator
            generator.load_state_dict(add_module_str_in_state_dict(torch.load(config['training']['generator_pretrained_ckpt_file'])))
            print('pretrained generator loaded!')
            # load pretrained discriminator
            pretrained_discriminator_loaded_dict = torch.load(config['training']['discriminator_pretrained_ckpt_file'])['discriminator']
            discriminator_state_dict = discriminator.state_dict()
            new_dict = {k: v for k, v in pretrained_discriminator_loaded_dict.items() if k not in ['module.fc.weight', 'module.fc.bias']}
            discriminator_state_dict.update(new_dict)
            discriminator.load_state_dict(discriminator_state_dict)
            print('pretrained discriminator loaded!')
        else:
            # load pretrained generator
            pretrained_generator_loaded_dict = torch.load(config['training']['pretrain_ckpt_file'])['generator']
            generator_state_dict = generator.state_dict()
            if config['training']['change_generator_fc_layer']:
                new_dict = {k: v for k, v in pretrained_generator_loaded_dict.items() if k not in ['module.embedding.weight','module.fc.weight', 'module.fc.bias']}
            else:
                new_dict = {k: v for k, v in pretrained_generator_loaded_dict.items() if k != 'module.embedding.weight'}
            if config['training']['special_class_embedding'] is not None:
                print('using special class embedding!')
                class_embedding = torch.FloatTensor(np.load(config['training']['special_class_embedding']))
                # (256,) -> (1,256)
                class_embedding = torch.unsqueeze(class_embedding, dim=0)
                new_dict['module.embedding.weight'] = class_embedding
            generator_state_dict.update(new_dict)
            generator.load_state_dict(generator_state_dict)
            print('pretrained generator loaded!')
            # load pretrained discriminator
            pretrained_discriminator_loaded_dict = torch.load(config['training']['pretrain_ckpt_file'])['discriminator']
            discriminator_state_dict = discriminator.state_dict()
            new_dict = {k: v for k, v in pretrained_discriminator_loaded_dict.items() if k not in ['module.fc.weight', 'module.fc.bias']}
            discriminator_state_dict.update(new_dict)
            discriminator.load_state_dict(discriminator_state_dict)
            print('pretrained discriminator loaded!')
    else:
        loaded_dict = torch.load(config['training']['pretrain_ckpt_file'])
        generator.load_state_dict(loaded_dict['generator'])
        discriminator.load_state_dict(loaded_dict['discriminator'])
        print('pretrained generator and discriminator loaded!')
    it = epoch_idx = -1

# Load checkpoint if it exists
try:
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    logger.load_stats('stats.p')
except FileNotFoundError:
    it = epoch_idx = -1

    


# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Evaluator
zdist_type=config['z_dist']['type']
evaluator = Evaluator(generator_test, zdist_type, zdist, ydist,
                      batch_size=batch_size, device=device)
                      
# Reinitialize model average if needed
if (config['training']['take_model_average']
        and config['training']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate anneling
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(
    generator, discriminator, g_optimizer, d_optimizer,
    gan_type=config['training']['gan_type'],
    reg_type=config['training']['reg_type'],
    reg_param=config['training']['reg_param'],
    frozen_generator=config['training']['frozen_generator'],
    frozen_discriminator=config['training']['frozen_discriminator'],
    frozen_generator_param_list=config['training']['frozen_generator_param_list'],
    frozen_discriminator_param_list=config['training']['frozen_discriminator_param_list']
)

# sample before training
print('Creating init samples...')
x = evaluator.create_samples(ztest, ytest)
logger.add_imgs(x, 'all', it)
for y_inst in range(sample_nlabels):
    x = evaluator.create_samples(ztest, y_inst)
    logger.add_imgs(x, '%04d' % y_inst, it)


# Training loop
print('Start training...')
flag = True
best_fid = np.infty
while flag:
    epoch_idx += 1
    print('Start epoch %d...' % epoch_idx)

    # decide wheather to use gmm when zdist type is gmm2gauss
    if zdist_type == 'gmm2gauss':
        if epoch_idx < max_epoch/2:
            use_gmm = True
        else:
            use_gmm = False

    for x_real, y in train_loader:
        it += 1
        
        

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'discriminator', d_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # Discriminator updates
        if config['z_dist']['type'] == 'gmm2gauss':
            z = zdist.sample((batch_size,), use_gmm)
        else:
            z = zdist.sample((batch_size,))
        dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
        d_scheduler.step()
        logger.add('losses', 'discriminator', dloss, it=it)
        logger.add('losses', 'regularizer', reg, it=it)

        # Generators updates
        if ((it + 1) % d_steps) == 0:
            if config['z_dist']['type'] == 'gmm2gauss':
                z = zdist.sample((batch_size,), use_gmm)
            else:
                z = zdist.sample((batch_size,))
            gloss = trainer.generator_trainstep(y, z)
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])
        g_scheduler.step()
        # Print stats
        g_loss_last = logger.get_last('losses', 'generator')
        d_loss_last = logger.get_last('losses', 'discriminator')
        d_reg_last = logger.get_last('losses', 'regularizer')
        print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
              % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last))

        # (i) Sample if necessary
        if (it % config['training']['sample_every']) == 0:
            print('Creating samples...')
            x = evaluator.create_samples(ztest, ytest)
            logger.add_imgs(x, 'all', it)
            for y_inst in range(sample_nlabels):
                x = evaluator.create_samples(ztest, y_inst)
                logger.add_imgs(x, '%04d' % y_inst, it)

        # (ii) Compute inception or fid if necessary
        if inception_every > 0 and ((it + 1) % inception_every) == 0:
            print('Computing inception score...')
            if config['z_dist']['type'] == 'gmm2gauss':
                inception_mean, inception_std = evaluator.compute_inception_score(use_gmm)
            else:
                inception_mean, inception_std = evaluator.compute_inception_score()
            logger.add('inception_score', 'mean', inception_mean, it=it)
            logger.add('inception_score', 'stddev', inception_std, it=it)

        if fid_every > 0 and ((it+1) % fid_every) == 0:
            # generate and save fake images
            print('Generating fake images to compute fid...')
            fid_fake_image_save_dir=os.path.join(out_dir, 'imgs','fid_fake_imgs')
            if config['z_dist']['type'] == 'gmm2gauss':
                evaluator.save_samples(sample_num=fid_fake_imgs_num, save_dir=fid_fake_image_save_dir, use_gmm=use_gmm)
            else:
                evaluator.save_samples(sample_num=fid_fake_imgs_num, save_dir=fid_fake_image_save_dir)
            print('Computiong fid...')
            fid_img_size = (config['data']['img_size'], config['data']['img_size'])
            fid = evaluator.compute_fid_score(generated_img_path = fid_fake_image_save_dir, 
                                                gt_path = config['data']['test_dir'] + '/0/', 
                                                img_size = fid_img_size)
            logger.add('fid', 'score', fid, it=it)
            if fid < best_fid:
                checkpoint_io.save('model_best.pt' , it=it)
                best_fid = fid

        # (iii) Backup if necessary
        if ((it + 1) % backup_every) == 0:
            print('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % it, it=it)
            logger.save_stats('stats_%08d.p' % it)

        # (iv) Save checkpoint if necessary
        if time.time() - t0 > save_every:
            print('Saving checkpoint...')
            checkpoint_io.save(model_file, it=it)
            logger.save_stats('stats.p')
            t0 = time.time()

            if (restart_every > 0 and t0 - tstart > restart_every):
                exit(3)
    if epoch_idx > max_epoch:
        flag = False
    if it > max_iter:
        flag = False