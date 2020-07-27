#!/bin/bash
export HDF5_USE_FILE_LOCKING='FALSE'
python train.py \
--dataset Lung --parallel --shuffle  --num_workers 16 --batch_size 96 --load_in_mem  \
--loss_type GAN \
--num_G_accumulations 4 --num_D_accumulations 4 \
--num_D_steps 1 --num_G_steps 1 --E_lr 1e-3 --G_lr 8e-4 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_init ortho --D_init ortho \
--G_eval_mode \
--dim_z 256 --shared_dim 256 --G_eval_mode --G_ch 64 --D_ch 64 \
--ema --use_ema --ema_start 20000 \
--test_every 4000 --save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 2023 \
--use_multiepoch_sampler
