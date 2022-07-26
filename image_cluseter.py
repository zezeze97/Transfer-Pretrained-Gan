import argparse
import os
from os import path
from re import L
from sys import implementation
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch import nn
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import shutil

# Arguments
parser = argparse.ArgumentParser(
    description='Run image clustering method.'
)
parser.add_argument('--data_path', type=str, help='Path to data')
parser.add_argument('--n_clusters', type=int, help='Num of clusters')
parser.add_argument('--output_dir', type=str, help='Path to save clustering result')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()


# Use gpu
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda:0" if is_cuda else "cpu")

# Build image feature extractor
image_feature_extractor =  models.resnet50(pretrained=True).to(device)
return_nodes = {'flatten': 'layer_last' }
# return_nodes = get_graph_node_names(image_feature_extractor)
image_feature_extractor = create_feature_extractor(image_feature_extractor, return_nodes=return_nodes).to(device)
image_feature_extractor.eval()

# Dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Get img feature
img_name_list = []
feature_mat = []
for img_name in tqdm(os.listdir(args.data_path)):
    img_name_list.append(img_name)
    img_path = os.path.join(args.data_path, img_name)
    img = transform(Image.open(img_path)).unsqueeze(0)
    with torch.no_grad():
        feature = image_feature_extractor(img.to(device))['layer_last']
        feature = feature.detach().cpu().numpy()
        feature_mat.append(feature)

feature_mat = np.concatenate(feature_mat)

# Kmeans Cluster
kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(feature_mat)
pred_cluster = kmeans.predict(feature_mat)

# Create new data 
if not os.path.exists(args.output_dir):
    for i in range(args.n_clusters):
        os.makedirs(os.path.join(args.output_dir, str(i)))



for img_id, cls_id in enumerate(pred_cluster):
    src_path = os.path.join(args.data_path, img_name_list[img_id])
    target_path = os.path.join(args.output_dir, str(cls_id), img_name_list[img_id])
    shutil.copy(src_path, target_path)


