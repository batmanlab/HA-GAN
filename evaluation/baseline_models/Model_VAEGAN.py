import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv3d(1, channel//16, kernel_size=4, stride=2, padding=1) # out:64
        self.conv2 = nn.Conv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # 32
        self.bn2 = nn.BatchNorm3d(channel//8)
        self.conv3 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1) # 16
        self.bn3 = nn.BatchNorm3d(channel//4)
        self.conv4 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # 8
        self.bn4 = nn.BatchNorm3d(channel//2)
        self.conv5 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # 4
        self.bn5 = nn.BatchNorm3d(channel)
        self.conv6 = nn.Conv3d(channel, out_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, h):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), negative_slope=0.2)
        h = self.conv6(h)
        h = torch.sigmoid(h.squeeze())
        return h

class Encoder(nn.Module):
    def __init__(self, channel=512, latent_size=1000):
        super(Encoder, self).__init__()
        
        self.latent_size = latent_size
        
        self.conv1 = nn.Conv3d(1, channel//16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//8)
        self.conv3 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//4)
        self.conv4 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel//2)
        self.conv5 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(channel)
        self.conv6 = nn.Conv3d(channel, channel, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm3d(channel)

        self.mean = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_size))

        self.logvar = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_size))
        
    def forward(self, h, _return_activations=False):

        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn6(self.conv6(h)), negative_slope=0.2)
        batch_size = h.size(0)
        h = h.view(batch_size, -1)
        
        mean = self.mean(h)
        logvar = self.logvar(h)

        std = logvar.mul(0.5).exp_()
        reparametrized_noise = torch.randn((batch_size, self.latent_size)).cuda()
        reparametrized_noise = mean + std * reparametrized_noise
        return mean, logvar, reparametrized_noise
    
class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel
        
        self.noise = noise
        self.fc = nn.Linear(1000,512*2*2*2)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv1 = nn.Conv3d(_c*8, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv3 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv4 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv5 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(_c)
        
        self.tp_conv6 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, h):
        h = h.view(-1, self.noise)
        h = self.fc(h)
        h = h.view(-1,512,2,2,2)
        h = F.relu(self.bn1(h))
        
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv1(h)
        h = F.relu(self.bn1(h))
        
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))
    
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))
        
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv5(h)
        h = F.relu(self.bn5(h))
        
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)
        
        h = torch.tanh(h)
        
        return h
