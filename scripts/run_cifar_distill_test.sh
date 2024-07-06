#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
source  activate ctdk
pwd
cd /home/23r9802_chen/CTDK/

python3 train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
        --distill kd \
        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --batch_size 64 --learning_rate 0.05 \
        --have_mlp 1 --mlp_name 'global' \
        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
        --save_model \
        --experiments_dir 'tea-res56-stu-res20/kd/global_T/your_experiment_name' \
        --experiments_name 'fold-1'
        
