# 预训练模型
- [lsun_bedroom](https://s3.eu-central-1.amazonaws.com/avg-projects/gan_stability/models/lsun_bedroom-df4e7dd2.pt): 256*256
- [imagenet](https://s3.eu-central-1.amazonaws.com/avg-projects/gan_stability/models/imagenet-8c505f47.pt): 128*128
  
    对应的预训练模型config在[configs/pretrained](configs/pretrained)中

# 数据集准备
    ```
    data
    └── LSUN
        ├── bedroom_train_lmdb
        ├── bedroom_val_lmdb
        ├── kitchen_minitrain_png
        ├── kitchen_train_lmdb
        ├── kitchen_val_lmdb
        └── kitchen_val_png
    ```

# Finetuning
 将所有Finetuning相关的模型config放在[configs/finetune](configs/finetune)中
## Bedroom -> Kitchen
    ```
    # 标准正态Noise
    python train.py configs/finetune/finetune_lsun_kitchen.yaml
    ```

# 隐向量寻找
    将所有隐向量寻找相关模型config放在[configs/img2vec2img](configs/img2vec2img)中
## 预训练Bedroom Generator中找Kitchen
    ```
    # 拟合kitchen 5w
    python img2vec2img.py configs/img2vec2img/img2vec2img_lsun_kitchen.yaml
    # 一个batch,一个batch拟合
    python img2vec2img_batch_mode.py configs/img2vec2img/img2vec2img_lsun_kitche_batch_mode.yaml
    ```

# 计算fid
    ```
    python test.py PATH_TO_CONFIG
    ```

to do:
在train.py中加入shift noise, 直接改zdist