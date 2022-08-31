# 预训练模型
- [lsun_bedroom](https://s3.eu-central-1.amazonaws.com/avg-projects/gan_stability/models/lsun_bedroom-df4e7dd2.pt): 256*256
- [imagenet](https://s3.eu-central-1.amazonaws.com/avg-projects/gan_stability/models/imagenet-8c505f47.pt): 128*128
  
    对应的预训练模型config在[configs/pretrained](configs/pretrained)中

# 数据集准备
    ```
        ├──data
            ├── cars1000cls5
            ├── cars_1000_sub
            ├── carscls5
            ├── cars_train
            ├── cathedral
            ├── cathedral_1000_sub
            ├── cathedral_25_sub
            ├── cityscapes
            ├── Flower25_cls5
            ├── Flowers
            ├── Flowers_1000_sub
            ├── Flowers_1000_sub_cls
            ├── Flowers_1000_sub_cls4
            ├── Flowers_25
            ├── Flowers251
            ├── Flowers_class
            ├── Flowers_cls
            ├── Imagenet
            ├── imagenet_fake_data
            ├── LSUN
            └── Oxford-IIIT-Pet
    ```

# 图片聚类
  在需要进行多类别(nlabels > 1)训练时，需要提前离线进行图片聚类

  ```
  python image_cluster.py --data_path {Path to data} --n_clusters {Num of clusters} --output_dir {Path to save clustering result}
  ```


# 隐向量寻找
  隐向量寻找的config放在[configs/vec2img](configs/vec2img)中

    ```
    python vec2img.py {path of vec2img config}
    eg: python vec2img.py configs/vec2img/flowers/vec2img_flowers25_cls5_special_class_embedding.yaml
    ```

# 隐空间建模
 将所有Finetuning相关的模型config放在[configs/finetune](configs/finetune)中的prefix，再进行gmm建模

 ```
 python latentvecs_modeling.py
 ```

# Finetuning
    ```
    # 单类别/多类别 gmm/gauss/shift gauss/kde
    python train.py {Path of finetuning config}

    # learnable gmm
    python train_learnable_gmm.py {Path of finetuning config}

    # class interpolate + contrast loss
    python train_class_interpolate.py {Path of finetuning config}

    # bss loss
    python train_bss.py {Path of finetuning config}

    # auxiliary sample + mmd loss
    python train_limited_data.py {Path of finetuning config}

    # tensorboard 可视化
    tensorborad --logdir {Path of output dir}
    ```

# 计算fid
  通常来说，看训练过程中的fid曲线即可，如果要测试，可使用如下命令
  
    ```
    # 不在imagenet上测试
    python test.py --config {Path of finetuning config}

    # 在imagenet上测试
    python test.py --config {Path of finetuning config} --test_imagenet --pretrained_ckpt_path {Path to pretrained ckpt if need to test imagenet} --imagenet_path {Path to imagenet}  
    
    ```
