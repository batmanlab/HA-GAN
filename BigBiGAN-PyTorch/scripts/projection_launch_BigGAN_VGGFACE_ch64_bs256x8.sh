#!/bin/bash
export HDF5_USE_FILE_LOCKING='FALSE'
python train.py \
--dataset VGGFACE2000 --parallel --shuffle  --num_workers 16 --batch_size 256 --load_in_mem  \
--loss_type Projection \
--num_G_accumulations 2 --num_D_accumulations 2 \
--num_D_steps 2 --num_G_steps 1 --G_lr 2e-4 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 64 --G_ch 64 \
--ema --use_ema --ema_start 20000 \
--test_every 4000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2018 \
--use_multiepoch_sampler
