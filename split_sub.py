import os
import shutil
import random

num_of_sample = 1000
image_root_path = 'data/Oxford-IIIT-Pet/0'
dst_root_path = 'data/Pet-Sub-1000/0'
all_image_list = os.listdir(image_root_path)
sample_image_list = random.sample(all_image_list, num_of_sample)

if not os.path.exists(dst_root_path):
    os.makedirs(dst_root_path)

for img_name in sample_image_list:
    shutil.copy(os.path.join(image_root_path, img_name), os.path.join(dst_root_path, img_name))
