# !/bin/bash
srun --job-name=ht -p GPU36 --nodes 1 --gres=gpu:1 --qos low --time 120:00:00 \
    python -u train.py configs/finetune/finetune_lsun_kitchen_baseline.yaml





