from tabnanny import verbose
import torch

pretraned_ckpt_path = 'pretrained_ckpt/imagenet/imagenet-8c505f47.pt'
vec2img_generator_path = 'output/vec2img/cityscapes_256dim_special_init_fix/chkpts/generator.pth'
device = torch.device("cpu")

pretrained_generator = torch.load(pretraned_ckpt_path, map_location=device)['generator']
vec2img_generator = torch.load(vec2img_generator_path, map_location=device)

for key in vec2img_generator.keys():
    print(key, (pretrained_generator['module.'+key]==vec2img_generator[key]).all())
