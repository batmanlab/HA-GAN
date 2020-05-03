# 3D HA-GAN


3D HA-GAN for 3D image generation.


## Usage

### Training

For unconditional COPD image generation (256,256,256):

python Alpha_GAN_COPD_train_256_SN_crop_v3plus_5_l1_GN.py

I typically run on 2 P100 or V100 GPUs, after 80000 iterations (~1.5 days), it produces resaonable result.

Use Tensorboard to monitor your training progress.

### Main model

Model_Alpha_GAN_256_SN_GN_v3.py

### Dataset class

COPD_dataset_slim.py


Brain_dataset_GSP_slim.py (For brain experiments)
