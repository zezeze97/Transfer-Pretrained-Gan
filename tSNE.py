"""
t-SNE: Visualize the distribution.
"""
import os
from time import time

import cv2
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
from gan_training.distributions import get_zdist
# NUM_PER_CLASS = 300
SEED = 12345


def get_data(data_path):
    """ Image data. """
    # transform_test = None
    # testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    img_paths = os.listdir(data_path)
    img_paths = [os.path.join(data_path, img_path) for img_path in img_paths]
    num_samples = len(img_paths)

    shape = (64, 64, 3)
    num_features = shape[0] * shape[1] * shape[2]
    samples = np.zeros((num_samples, num_features))

    for i in range(num_samples):
        img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
        img = cv2.resize(img, shape[:-1]).flatten() / 255.
        samples[i, :] = img

    # from sklearn.preprocessing import StandardScaler
    # samples_std = StandardScaler().fit_transform(samples)
    # return samples, num_samples, num_features
    return samples

def load_latentvecs(latentvec_dir, num):


    # load latent vectors npy file
    for i,filename in enumerate(os.listdir(latentvec_dir)):
        if i == 0:
            latent_vecs = np.load(latentvec_dir + filename)
        else:
            current_vecs = np.load(latentvec_dir + filename)
            latent_vecs = np.concatenate((current_vecs,latent_vecs),axis=0)
        total_num = latent_vecs.shape[0]
    return latent_vecs[total_num-num:,:]

# total_feat_in = []
#
# def hook_fn_forward(module, input, output):
#     total_feat_in.append(input)
#
# def get_featuremap_data(data_path):
#     """ Feature maps of test samples. """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     net = resnet.ResNet50()
#     net = net.to(device)
#     ckpt_path = './checkpoint/resnet50.pth'
#     state_dict = torch.load(ckpt_path)
#     net.load_state_dict(state_dict['net'])
#     for name, layer in net.named_modules():
#         if name == 'linear':  # hook the input of fc layer
#             layer.register_forward_hook(hook_fn_forward)
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     testset = torchvision.datasets.ImageFolder(root=data_path, transform=None)
#     total_len = len(testset)
#     num_class = len(testset.classes)  # 10
#     # shape = np.array(testset[0][0]).shape  # 96x96x3 when transform is None
#     print(np.array(testset[0][0]).shape)
#     num_samples = num_class * NUM_PER_CLASS
#     num_features = 2048
#     samples = np.zeros((num_samples, num_features))
#     labels = [0] * num_samples
#     count = {}
#     for i in range(num_class):
#         count[i] = 0
#     n_samp = 0
#
#     testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
#     with torch.no_grad():
#         for i in tqdm(range(total_len)):
#             sample, label = testset[i]
#             sample = sample.to(device)
#             if count[label] < NUM_PER_CLASS:
#                 count[label] += 1
#                 sample = sample.unsqueeze(0)
#                 net(sample)
#                 feature_map = total_feat_in[0][0].squeeze(0)
#                 # feature_map = net.last_feature_map.squeeze(0)
#                 # print(feature_map.size())
#                 samples[n_samp, :] = feature_map.cpu().numpy()  # .flatten()
#                 labels[n_samp] = label
#                 n_samp += 1
#
#                 total_feat_in.clear()
#
#     return samples, labels, num_samples, num_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main(type):
    if type == 'image':
        data_path = 'data/kitchen_val_png/0'
        gt_samples = get_data(data_path)   
        data_path = 'outputs/generate_results/kitchen_finetune_300'
        shift_samples = get_data(data_path)
        data_path = 'outputs/generate_results/kitchen_finetune_randnoise_300'
        randn_samples = get_data(data_path)
        all_samples = np.concatenate((gt_samples,shift_samples,randn_samples),axis=0)
        print(all_samples.shape)
        print('Computing t-SNE embedding')
        # TSEN default: perplexity=30, n_iter=1000
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)  # init='pca'
        t0 = time()
        all_results = tsne.fit_transform(all_samples)
        print('time: %.2f' % (time() - t0))
        gt_result = all_results[:300,]
        plt.scatter(x=gt_result[:, 0], y=gt_result[:, 1], c='r', s=1, label='real')
        shift_result = all_results[300:600,:]
        plt.scatter(x=shift_result[:, 0], y=shift_result[:, 1], c='g', s=1, label='shift_noise_generate')
        randn_result = all_results[600:,:]
        plt.scatter(x=randn_result[:, 0], y=randn_result[:, 1], c='b', s=1, label='randn_noise_generate')

        plt.title('kitchen_t-SNE')
        plt.legend()  # loc='upper left'
        plt.savefig('outputs/generate_results/kitchen_t-SNE.png')
    if type == 'latentvecs':
        latentvec_dir = 'output/img2vec2img/lsun_kitchen_batch_mode/latentvecs/'
        num = 1000
        samples_kitchen = load_latentvecs(latentvec_dir, num)  
        zdist = get_zdist('gauss', 256, mean=None, cov=None, device='cpu')
        samples_bedroom = zdist.sample((1000,)).numpy() 
        all_samples = np.concatenate((samples_kitchen,samples_bedroom),axis=0)
        print(all_samples.shape)
        print('Computing t-SNE embedding')
        # TSEN default: perplexity=30, n_iter=1000
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)  # init='pca'
        t0 = time()
        all_results = tsne.fit_transform(all_samples)
        print('time: %.2f' % (time() - t0))
        # plot_embedding(results, labels, 't-SNE embedding of images')
        kitchen_result = all_results[:1000,:]
        plt.scatter(x=kitchen_result[:, 0], y=kitchen_result[:, 1], c='g', s=1, label='kitchen_latent_vecs')
        bedroom_result = all_results[1000:,:]
        plt.scatter(x=bedroom_result[:, 0], y=bedroom_result[:, 1], c='r', s=1, label='bedroom_latent_vecs')

        plt.title('latentvecs_t-SNE')
        plt.legend()  # loc='upper left'
        plt.savefig('output/img2vec2img/lsun_kitchen_batch_mode/latentvecs_t-SNE.png')





if __name__ == '__main__':
    type = 'latentvecs'
    main(type)
