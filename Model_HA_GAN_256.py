import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from layers import SNConv3d, SNLinear

class Code_Discriminator(nn.Module):
    def __init__(self, code_size, num_units=256):
        super(Code_Discriminator, self).__init__()

        self.l1 = nn.Sequential(SNLinear(code_size, num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(SNLinear(num_units, num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = SNLinear(num_units, 1)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)

        return h3

class Sub_Encoder(nn.Module):
    def __init__(self, channel=256, n_class=1):
        super(Sub_Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(channel//4, channel//8, kernel_size=4, stride=2, padding=1) # in:[64,64,64], out:[32,32,32]
        self.bn1 = nn.GroupNorm(8, channel//8)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1) # out:[16,16,16]
        self.bn2 = nn.GroupNorm(8, channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # out:[8,8,8]
        self.bn3 = nn.GroupNorm(8, channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[4,4,4]
        self.bn4 = nn.GroupNorm(8, channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0) # out:[1,1,1,1]

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu(self.bn1(h))
        h = self.conv2(h)
        h = self.relu(self.bn2(h))
        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        h = self.conv4(h)
        h = self.relu(self.bn4(h))
        h = self.conv5(h).squeeze()
        return h

class Encoder(nn.Module):
    def __init__(self, channel=64):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, channel//2, kernel_size=4, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
        self.bn1 = nn.GroupNorm(8, channel//2)
        self.conv2 = nn.Conv3d(channel//2, channel//2, kernel_size=3, stride=1, padding=1) # out:[16,128,128]
        self.bn2 = nn.GroupNorm(8, channel//2)
        self.conv3 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[8,64,64]
        self.bn3 = nn.GroupNorm(8, channel)

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))

        h = self.conv2(h)
        h = self.relu(self.bn2(h))

        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        return h

class Sub_Discriminator(nn.Module):
    def __init__(self, channel=256):
        super(Sub_Discriminator, self).__init__()
        self.channel = channel
        n_class = 1

        self.conv1 = SNConv3d(1, channel//8, kernel_size=4, stride=2, padding=1) # in:[64,64,64], out:[32,32,32]
        self.conv2 = SNConv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1) # out:[16,16,16]
        self.conv3 = SNConv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # out:[8,8,8]
        self.conv4 = SNConv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[4,4,4]
        self.conv5 = SNConv3d(channel, n_class, kernel_size=4, stride=1, padding=0) # out:[1,1,1,1]

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
        
        self.conv1 = SNConv3d(1, channel//32, kernel_size=4, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
        self.conv2 = SNConv3d(channel//32, channel//16, kernel_size=4, stride=2, padding=1) # out:[8,64,64,64]
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[4,32,32,32]
        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[2,16,16,16]
        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[1,8,8,8]
        self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,4,4,4]
        self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,4,4), stride=1, padding=0) # out:[1,1,1,1]
        self.fc1 = SNLinear(channel//4+1, channel//8)
        self.fc2 = SNLinear(channel//8, n_class)
        self.sub_D = Sub_Discriminator()

    def forward(self, h, h_small, crop_idx, from_E=False):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze()
        h = torch.cat([h, (crop_idx / 224. * torch.ones((h.size(0), 1))).cuda()], 1) # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h = self.fc2(h)
        if not from_E:
            h_small = self.sub_D(h_small)
            return (h + h_small) / 2.
        return h

class Sub_Generator(nn.Module):
    def __init__(self, channel:int=16):
        super(Sub_Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*2)

        self.tp_conv2 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c)

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
    def __init__(self, noise:int=1000, channel:int=32, mode="train"):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.noise = noise

        self.fc1 = nn.Linear(noise, 4*4*4*_c*16)

        self.tp_conv1 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c*2)

        self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.sub_G = Sub_Generator(channel=_c//2)
 
    def forward(self, h, crop_idx, return_latent=False, latent_only=False):

        if crop_idx != None:
            h = self.fc1(h)
            h = h.view(-1,512,4,4,4)
            h = self.tp_conv1(h)
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
            h_latent = self.relu(self.bn5(h)) # (64, 64, 64), channel:128
            if latent_only:
                return h_latent
            h_small = self.sub_G(h_latent)
            h = h_latent[:,:,crop_idx//4:crop_idx//4+8,:,:] # Crop, out: (8, 64, 64)

        h1 = F.interpolate(h,scale_factor = 2)
        h1 = self.tp_conv6(h1)
        h1 = self.relu(self.bn6(h1)) # (128, 128, 128)

        h1 = F.interpolate(h1,scale_factor = 2)
        h1 = self.tp_conv7(h1)

        h1 = torch.tanh(h1) # (256, 256, 256)

        if crop_idx != None:
            if return_latent:
                return h1, h_small, h_latent
            return h1, h_small
        return h1
