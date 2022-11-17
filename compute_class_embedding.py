import argparse
import os
from os import path
import torch
from torch import nn
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_generator
from gan_training.distributions import get_ydist, get_zdist
from collections import OrderedDict
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
import torchvision.models as models
import json
import matplotlib.pyplot as plt 
import pandas as pd

def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename


# load imagenet class info 
with open('imagenet_class_index.json','r')as f:
    imagenet_class = json.load(f)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
# parser.add_argument('--outdir', type=str, help='Path to save class_embedding.')
parser.add_argument('--softlabel', action='store_true', help='use logitic.')
args = parser.parse_args()

config = load_config(args.config, default_path=None)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Short hands
batch_size = config['training']['batch_size']
out_dir = path.join('output',config['training']['class_embedding'].split('/')[1])
class_embedding_dir = path.join(out_dir, 'class_embedding')

# Create missing directories
if not path.exists(class_embedding_dir):
    os.makedirs(class_embedding_dir)


# use gpu
device = torch.device("cuda:0" if is_cuda else "cpu")

# Dataset
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    pretrained_transform=True
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=False, pin_memory=True, sampler=None, drop_last=True
)


# Number of labels
config['data']['nlabels'] = 1000


# Create models
generator = build_generator(config)
label_encoder = models.resnet50(pretrained=True)
softmax = nn.Softmax(dim=1)
label_dim = config['generator']['kwargs']['embed_size']




# Put models on gpu if needed
generator = generator.to(device)
label_encoder = label_encoder.to(device)




# Load generator ckpt
pretrained_ckpt = config['training']['pretrain_ckpt_file']
loaded_dict = torch.load(pretrained_ckpt)
print('Loading pretrained generator...')
generator.load_state_dict(remove_module_str_in_state_dict(loaded_dict))
print('Pretrained generator loaded!')






# fix param in generator
for params in generator.parameters():
    params.requires_grad = False

# fix param in label_encoder 
for params in label_encoder.parameters():
    params.requires_grad = False

# set mode
generator.eval()
label_encoder.eval()

# get label_embedding_layer_weights 
label_embedding_layer_weights = generator.embedding.weight.clone()

# compute label_embedding 
print('computing label embedding...')

label_embedding = []
pred_label_index = []
for (x_real, y) in train_loader:
    with torch.no_grad():
        logitics = label_encoder(x_real.to(device))
        label_index = torch.argmax(logitics, dim=1)
        pred_label_index.append(label_index.detach().cpu().tolist())
        if args.softlabel:
            label_embedding_dist = softmax(logitics)
            label_embedding.append(torch.mm(label_embedding_dist, label_embedding_layer_weights))
        else:
            label_embedding.append(generator.embedding(label_index))
label_embedding = torch.mean(torch.concat(label_embedding),dim=0).squeeze()
pred_label_index = np.array(pred_label_index).flatten()

pred_label = []
for index in pred_label_index:
    pred_label.append(imagenet_class[str(index)][1])

print('finish computing label_embedding!!')



# save label_embedding
label_embedding_np = label_embedding.detach().cpu().numpy()
np.save(out_dir + '/class_embedding/class_embedding.npy', label_embedding_np)

# visualize currrent label embedding
zdist = get_zdist(dist_name='gauss',dim=config['z_dist']['dim'], device=device)

z = zdist.sample((batch_size,))
with torch.no_grad():         
    y = torch.zeros((batch_size, label_dim)).to(device)
    for i in range(batch_size):
        y[i,:] = label_embedding
    gen_images = generator(z, y)
gen_images = torchvision.utils.make_grid(gen_images*0.5+0.5, nrow=8, padding=2)
torchvision.utils.save_image(gen_images, path.join(class_embedding_dir,'gen_images.png'))

# visualize pred_label
se = pd.Series(pred_label)
countDict = dict(se.value_counts())
proportitionDict = dict(se.value_counts(normalize=True))
simplify_countDict = {}
cumulative_count = 0
for k,v in proportitionDict.items():
    if v > 0.03:
        simplify_countDict[k] = countDict[k]
        cumulative_count += countDict[k]
simplify_countDict['else'] = len(pred_label) - cumulative_count

plt.bar(list(simplify_countDict.keys()), simplify_countDict.values(), color='b')  
plt.savefig(path.join(out_dir, "pred_label_hist.png"))
    