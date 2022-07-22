import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser(
    description='Compute coressponding angle in transfering gan.'
)
parser.add_argument('--finetune_ckpt_path_baseline', type=str, help='Path to finetune config file.')
parser.add_argument('--finetune_ckpt_path', type=str, help='Path to finetune config file.')
parser.add_argument('--pretrained_ckpt_path', type=str, help='Path to pretrained config file.')
parser.add_argument('--num_of_index',  type=int, help='max num of index to save.')
parser.add_argument('--outdir', type=str, help='Path to save output')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')


args = parser.parse_args()
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Set device
device = torch.device("cuda:0" if is_cuda else "cpu")



# Load checkpoint 
finetnue_base_ckpt_stat_dict = torch.load(args.finetune_ckpt_path_baseline, map_location=device)
finetune_ckpt_stat_dict = torch.load(args.finetune_ckpt_path, map_location=device)
pretrained_ckpt_stat_dict = torch.load(args.pretrained_ckpt_path, map_location=device)
finetune_generator_base_weights = finetnue_base_ckpt_stat_dict['generator']
finetune_generator_weights = finetune_ckpt_stat_dict['generator']
pretrained_generator_weights = pretrained_ckpt_stat_dict['generator']

# Only conv weigh take into account!
layer_list = []
for k in finetune_generator_weights.keys():
    if 'conv' in k:
        if 'weight' in k:
            layer_list.append(k)

# cos sim
def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  
    return np.abs(num / denom) if denom != 0 else 0

select_layer_list = ["module.resnet_0_0.conv_0.weight",
                        "module.resnet_1_0.conv_0.weight",
                        "module.resnet_2_0.conv_0.weight",
                        "module.resnet_3_0.conv_0.weight",
                        "module.resnet_4_0.conv_0.weight",
                        "module.resnet_5_0.conv_0.weight",
                        "module.conv_img.weight"]

# select_layer_list = ["module.resnet_0_0.conv_0.weight","module.resnet_0_1.conv_0.weight"]

output_dict = {}
for key in tqdm(select_layer_list):
    fintune_g_base_weight_mat = finetune_generator_base_weights[key]
    fintune_g_weight_mat = finetune_generator_weights[key]
    pretrained_g_weight_mat = pretrained_generator_weights[key]
    assert fintune_g_weight_mat.shape == pretrained_g_weight_mat.shape
    ori_shape = fintune_g_weight_mat.shape
    fintune_g_base_weight_mat = fintune_g_base_weight_mat.view(ori_shape[0],-1).cpu().numpy()
    fintune_g_weight_mat = fintune_g_weight_mat.view(ori_shape[0],-1).cpu().numpy()
    pretrained_g_weight_mat = pretrained_g_weight_mat.view(ori_shape[0],-1).cpu().numpy()
    # svd
    f_b_U, f_b_sigma, f_b_VT = np.linalg.svd(fintune_g_base_weight_mat)
    f_U, f_sigma, f_VT = np.linalg.svd(fintune_g_weight_mat)
    p_U, p_sigma, p_VT = np.linalg.svd(pretrained_g_weight_mat)
    assert f_U.shape == p_U.shape
    sim_list = []
    sim_b_list = []
    for i in range(min(f_U.shape[1], args.num_of_index)):
        cos_sim = get_cos_similar(f_U[:,i], p_U[:,i])
        sim_list.append(cos_sim)
        cos_sim = get_cos_similar(f_b_U[:,i], p_U[:,i])
        sim_b_list.append(cos_sim)
    output_dict[key] = {'gauss':sim_b_list,'gmm':sim_list}

# save
outdir = os.path.join(args.outdir,args.finetune_ckpt_path.split('/')[-3])
if not os.path.exists(outdir):
    os.makedirs(outdir)

output_file_path = os.path.join(outdir,"coresponding_angle.json")
with open(output_file_path,'w',encoding='utf-8') as f:
    json.dump(output_dict, f,ensure_ascii=False)

# draw image
for key in tqdm(select_layer_list):
    plt.figure(figsize=(10,5))
    plt.title("cos value of coressponding angle")
    plt.xlabel("index")
    plt.ylabel("Coressponding Angles")
    x = range(len(output_dict[key]['gauss']))
    plt.plot(x,output_dict[key]['gauss'], label="gauss."+key[7:])
    plt.plot(x,output_dict[key]['gmm'], label="gmm."+key[7:])
    plt.legend()
    output_fig_path = os.path.join(outdir,"coresponding_angle" + key[7:] + ".png")
    plt.savefig(output_fig_path)
