#!/bin/bash
export HDF5_USE_FILE_LOCKING='FALSE'
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--loss_type Twin_AC --AC \
--AC_weight 1.0 \
--shuffle --batch_size 100 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 2 --num_G_steps 1 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--test_every 4000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2020 \
--ema  --use_ema --ema_start 10000 \
