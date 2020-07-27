# Hierarchical Amortized GAN (HA-GAN)

Official PyTorch implementation for paper *Hierarchical Amortized Training for Memory-efficient High Resolution 3D GAN*.

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/volGAN-For-Lung/blob/master/figures/main.png">
</p>

### Requirements
- PyTorch 1.4
- scikit-image
- nibabel
- nilearn
- tensorboardX

### Training
```bash
python train.py
```

### Evaluation
```
evaluation
├── visualization_HA_GAN_COPD.ipynb - Notebook to visualize generated images on COPD dataset.
├── visualization_HA_GAN_Brain.ipynb - Notebook to visualize generated images on GSP dataset.
├── PCA_embedding_COPD.ipynb - Notebook to visualize PCA embedding of generated images on COPD dataset.
├── PCA_embedding_Brain_GSP.ipynb - Notebook to visualize PCA embedding of generated images on GSP dataset.
├── PCA_embedding_COPD.ipynb - Notebook to visualize PCA embedding of generated images on COPD dataset.
├── structure_wise_dice.ipynb - Notebook to visualize structure-wise reconstruction quality on GSP dataset (Dice).
└── structure_wise_hausdorff.ipynb - Notebook to visualize structure-wise reconstruction quality on GSP dataset (95% Hausdorff distance).
```
### Sample images
<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/volGAN-For-Lung/blob/master/figures/result_reconstruction.png">
</p>
