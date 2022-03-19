# !/bin/bash 

#srun --job-name=gan-gauss -p GPU36 --gres=gpu:1 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen.yaml

srun --job-name=gan-gauss -p GPU36 --gres=gpu:1 --qos low --time 120:00:00 \
    python -u train.py configs/finetune/finetune_lsun_kitchen_shift.yaml


