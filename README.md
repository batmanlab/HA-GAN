# Hierarchical Amortized GAN (HA-GAN)

Official PyTorch implementation for paper *Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis*.

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/HA-GAN/blob/master/figures/main_github.png">
</p>

#### [[Paper & Supplementary Material]](https://ieeexplore.ieee.org/abstract/document/9770375)

Generative Adversarial Networks (GAN) have many potential medical imaging applications. Due to the limited memory of Graphical Processing Units (GPUs), most current 3D GAN models are trained on low-resolution medical images. In this work, we propose a novel end-to-end GAN architecture that can generate high-resolution 3D images. We achieve this goal by using different configurations between training and inference. During training, we adopt a hierarchical structure that simultaneously generates a low-resolution version of the image and a randomly selected sub-volume of the high-resolution image. The hierarchical design has two advantages: First, the memory demand for training on high-resolution images is amortized among sub-volumes. Furthermore, anchoring the high-resolution sub-volumes to a single low-resolution image ensures anatomical consistency between sub-volumes. During inference, our model can directly generate full high-resolution images. We also incorporate an encoder (hidden in the figure to improve clarity) into the model to extract features from the images.

### Requirements
- PyTorch
- scikit-image
- nibabel
- nilearn
- tensorboardX

```bash
conda env create --name hagan -f environment.yml
```

### Data Preprocessing
```bash
python preprocess.py
```

### Training
#### Unconditional HA-GAN
```bash
python train.py --workers 8 --num-class 0 --exp-name 'HA_GAN_run1' --data-dir DATA_DIR
```
#### Conditional HA-GAN
```bash
python train.py --workers 8 --num-class N --exp-name 'HA_GAN_cond_run1' --data-dir DATA_DIR
```

Track your training with Tensorboard:
<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/HA-GAN/blob/master/figures/tensorboard.png">
</p>

### Testing
```bash
visualization.ipynb
python evaluation/fid_score.py
```

### Sample images
<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/HA-GAN/blob/master/figures/sample_HA_GAN.png">
</p>

### Pretrained model
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Iteration</th>
<th valign="bottom">Checkpoint</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000179.v6.p2">COPDGene</a></td>
<td align="center">80000</td>
<td align="center"><a href="https://drive.google.com/file/d/1orNvz7DLsCn5KWKjjVpEL4e5mO0akf6g/view?usp=sharing">download</a></td>
</tr>
  
</tbody></table>

### Citation
```
@ARTICLE{hagan2022,
  author={Sun, Li and Chen, Junxiang and Xu, Yanwu and Gong, Mingming and Yu, Ke and Batmanghelich, Kayhan},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JBHI.2022.3172976}}
```
