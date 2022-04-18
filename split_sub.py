import os
import shutil
import random

num_of_sample = 1000
image_root_path = 'data/cars_train/0'
dst_root_path = 'data/cars_1000_sub/0'
all_image_list = os.listdir(image_root_path)
sample_image_list = random.sample(all_image_list, num_of_sample)

for img_name in sample_image_list:
    shutil.copy(os.path.join(image_root_path, img_name), os.path.join(dst_root_path, img_name))
