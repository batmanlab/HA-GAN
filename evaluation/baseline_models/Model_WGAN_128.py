import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        n_class = 1
        
        self.conv1 = nn.Conv3d(1, channel//16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//8)
        self.conv3 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//4)
        self.conv4 = nn.Conv3d(channel//4, channel//2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn4 = nn.BatchNorm3d(channel//2)
        self.conv5 = nn.Conv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn5 = nn.BatchNorm3d(channel)
        self.conv6 = nn.Conv3d(channel, n_class, kernel_size=(1,4,4), stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)

        return h6
    

class Generator(nn.Module):
    def __init__(self, noise:int=1000, channel:int=64):
        super(Generator, self).__init__()
        _c = channel
        
        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*16, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(_c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*8)

        self.tp_conv3 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*4)

        self.tp_conv4 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c*2)

        self.tp_conv5 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(_c)

        self.tp_conv6 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):

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

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)

        h = torch.tanh(h) # (256, 256, 256)

        return h
