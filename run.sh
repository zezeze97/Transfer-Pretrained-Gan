# !/bin/bash
#srun --job-name=base -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_baseline.yaml
#srun --job-name=shift -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_shift_vec2img_batch_mode.yaml
#srun --job-name=gmm -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_gmm_vec2img_batch_mode.yaml

#srun --job-name=base -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_flowers_512dim_small_lr.yaml
srun --job-name=shuffle-autoshift -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    python -u train_autoshift.py configs/finetune/finetune_flowers_512dim_autoshift_shuffle_small_lr.yaml

