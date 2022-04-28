# !/bin/bash
#srun --job-name=base -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_baseline.yaml
#srun --job-name=shift -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_shift_vec2img_batch_mode.yaml
#srun --job-name=gmm -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_lsun_kitchen_gmm_vec2img_batch_mode.yaml

#srun --job-name=base -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train.py configs/finetune/finetune_flowers_512dim_small_lr.yaml
#srun --job-name=shuffle-autoshift -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train_autoshift.py configs/finetune/finetune_flowers_512dim_autoshift_shuffle_small_lr.yaml
#srun --job-name=save-autoshift -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train_autoshift_save.py configs/finetune/finetune_flowers_512dim_autoshift_save_shuffle_small_lr.yaml
#srun --job-name=dist-reg-sas -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train_autoshift_save.py configs/finetune/finetune_flowers_512dim_autoshift_save_shuffle_dist_reg_small_lr.yaml
#srun --job-name=256-save-autoshift -p GPU --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    #python -u train_autoshift_save.py configs/finetune/finetune_flowers_256dim_autoshift_save_shuffle_small_lr.yaml
srun --job-name=nB-sas -p GPU36 --nodes 1 --gres=gpu:1 --cpus-per-task 6 --qos low --time 120:00:00 \
    python -u train_autoshift_save_v1.py configs/finetune/finetune_flowers_512dim_autoshift_save_per_batch_shuffle_small_lr.yaml

