import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class Sub_Discriminator(nn.Module):
    def __init__(self, channel=256):
        super(Sub_Discriminator, self).__init__()
        self.channel = channel
        n_class = 1

        self.conv1 = spectral_norm(nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)) # in:[64,64,64], out:[32,32,32]
        self.conv2 = spectral_norm(nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)) # out:[16,16,16]
        self.conv3 = spectral_norm(nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)) # out:[8,8,8]
        self.conv4 = spectral_norm(nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)) # out:[4,4,4]
        self.conv5 = spectral_norm(nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)) # out:[1,1,1,1]

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = self.conv5(h).view((-1,1))
        return h

class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        n_class = 1
        
        self.conv1 = spectral_norm(nn.Conv3d(1, channel//32, kernel_size=4, stride=2, padding=1)) # in:[32,256,256], out:[16,128,128]
        self.conv2 = spectral_norm(nn.Conv3d(channel//32, channel//16, kernel_size=4, stride=2, padding=1)) # out:[8,64,64,64]
        self.conv3 = spectral_norm(nn.Conv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1)) # out:[4,32,32,32]
        self.conv4 = spectral_norm(nn.Conv3d(channel//8, channel//4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1))) # out:[2,16,16,16]
        self.conv5 = spectral_norm(nn.Conv3d(channel//4, channel//2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1))) # out:[1,8,8,8]
        self.conv6 = spectral_norm(nn.Conv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))) # out:[1,4,4,4]
        self.conv7 = spectral_norm(nn.Conv3d(channel, n_class, kernel_size=(1,4,4), stride=1, padding=0)) # out:[1,1,1,1]
        self.sub_D = Sub_Discriminator()

    def forward(self, h, h_small):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = self.conv7(h).view((-1,1))
        h_small = self.sub_D(h_small)
        return (h + h_small) / 2.

class Sub_Generator(nn.Module):
    def __init__(self, channel:int=32):
        super(Sub_Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(_c*2)

        self.tp_conv2 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(_c)

        self.tp_conv3 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h):

        h = self.tp_conv1(h)
        h = self.relu(self.bn1(h))

        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = self.tp_conv3(h)
        h = torch.tanh(h)
        return h

class Generator(nn.Module):
    def __init__(self, noise:int=1000, channel:int=64, mode="train"):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*16, kernel_size=4, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm3d(_c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(_c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(_c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(_c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(_c*2)

        self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(_c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.sub_G = Sub_Generator()
 
    def forward(self, noise, crop_idx):

        noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(noise)
        h = self.relu(self.bn1(h))

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv5(h)
        h = self.relu(self.bn5(h)) # (64, 64, 64)

        if self.mode == "train":
            h_small = self.sub_G(h)
            h = h[:,:,crop_idx//4:crop_idx//4+8,:,:] # Crop

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h)) # (128, 128, 128)

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv7(h)

        h = torch.tanh(h) # (256, 256, 256)

        if self.mode == "train":
            return h, h_small
        else:
            return h
