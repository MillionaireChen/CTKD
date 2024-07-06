#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
source  activate ctdk
pwd
cd /home/23r9802_chen/CTDK/

python3 train_teacher.py --model resnet56 \
                --batch_size 64 \
                --epochs 1 \
                --learning_rate 0.05 \
                --lr_decay_epochs '150,180,210' \
                --lr_decay_rate 0.1 \
                --experiments_dir 'baseline/resnet56' \
                --experiments_name 'fold-1'