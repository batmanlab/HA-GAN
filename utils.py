import math
from collections import OrderedDict

import numpy as np
from skimage.transform import resize

import torch

def post_process_brain(x_pred):
    x_pred = resize(x_pred, (256-90,256-40,256-40), mode='constant', cval=0.)
    x_canvas = np.zeros((256,256,256))
    x_canvas[50:-40,20:-20,20:-20] = x_pred
    x_canvas = np.flip(x_canvas,0)
    return x_canvas

def _itensity_normalize(volume):       
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    return out
    
class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

def calculate_nmse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    mse0 = np.mean(img1**2)
    if mse == 0:
        return float('inf')
    return mse / mse0 * 100.

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

class KLN01Loss(torch.nn.Module):

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), 'wtf?'

        samples = samples.view(samples.size(0), -1)

        self.samples_var = var(samples)
        self.samples_mean = samples.mean(0)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            # mu_1 = 0; sigma_1 = 1

            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # mu_2 = 0; sigma_2 = 1

            t1 = (samples_var.pow(2) + samples_mean.pow(2)) / 2
            t2 = -samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1

        return KL

def trim_state_dict_name(state_dict):
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    return state_dict

def inf_train_gen(data_loader):
    while True:
        for _,batch in enumerate(data_loader):
            yield batch
