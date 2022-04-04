# !/bin/bash
srun --job-name=ht -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    python -u train.py configs/finetune/finetune_lsun_kitchen_baseline.yaml

#srun --job-name=ht -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_shift_vec2img_batch_mode.yaml



