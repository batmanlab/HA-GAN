# 3D SubsampleGAN

3D Subsample GAN for 3D image generation


## Usage

### Training

For unconditional COPD image generation (256*256*256):

Latest development version:
python GAN_COPD_train_256_SN_crop_v6.py

A legacy but stable version:
python GAN_COPD_train_256_SN_crop_v5.py

I typically run on 2 P100 or V100 GPUs, after 80000 iterations (~1.5 days), it produces resaonable result.
Use Tensorboard to monitor your training progress.

For conditional BRATS image generation (160*240*240):
python GAN_BRATS_train_240_SN_crop_v3_pix2pix_l1.py

I typically run on 2 P100 or V100 GPUs, after 18500 iterations (~0.5 days), it produces resaonable result.

### Main model

Model_GAN_256_SN_v5.py

Model_GAN_256_SN_v6.py

Model_GAN_240_SN_v3plus_pix2pix.py

### Dataset class

BRATS_dataset_slim.py

COPD_dataset_slim.py
