GPU=$1


# === Finetune ===
#CUDA_VISIBLE_DEVICES=$GPU 
srun --job-name=finetune_lsun_kitchen_shift_batch_mode -p GPU36 --gres=gpu:1 --qos low --time 120:00:00 \
    python train.py configs/finetune/finetune_lsun_kitchen_shift_batch_mode.yaml



