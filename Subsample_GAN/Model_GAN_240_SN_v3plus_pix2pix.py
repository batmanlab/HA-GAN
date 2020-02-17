import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from layers import SNConv3d, SNLinear
 
class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        n_class = 1
        
        self.conv1 = SNConv3d(2, channel//32, kernel_size=4, stride=2, padding=1) # concat input,in:[32,240,240],out:[16,120,120]
        self.conv2 = SNConv3d(channel//32, channel//16, kernel_size=4, stride=2, padding=1) # out:[8,60,60]
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[4,30,30]
        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=4, stride=(1,2,2), padding=(0,1,1)) # out:[1,15,15]
        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) # out:[1,8,8]
        self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,4,4]
        self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,4,4), stride=1, padding=0) # out:[1,1,1]
        self.fc1 = SNLinear(channel//4+1, channel//8)
        self.fc2 = SNLinear(channel//8, n_class)

    def forward(self, x, crop_idx):
        h = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze()
        #print(h.size())
        h = torch.cat([h, (crop_idx / 128. * torch.ones((h.size(0), 1))).cuda()], 1) # 160-32
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h = self.fc2(h)
        return h

class Generator(nn.Module):
    def __init__(self, channel=64):
        super(Generator, self).__init__()
        _c = channel
        
        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(1, _c, kernel_size=4, stride=2, padding=1, bias=True) # in:[32,240,240], out:[16,120,120]
        self.bn1 = nn.BatchNorm3d(_c)

        self.tp_conv2 = nn.Conv3d(_c, _c*2, kernel_size=4, stride=2, padding=1, bias=True) # out:[8,60,60]
        self.bn2 = nn.BatchNorm3d(_c*2)

        self.tp_conv3 = nn.Conv3d(_c*2, _c*4, kernel_size=4, stride=2, padding=1, bias=True) # out:[4,30,30]
        self.bn3 = nn.BatchNorm3d(_c*4)

        self.tp_conv4 = nn.Conv3d(_c*4, _c*4, kernel_size=3, stride=1, padding=1, bias=True) # out:[4,30,30]
        self.bn4 = nn.BatchNorm3d(_c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True) # out:[8,60,60]
        self.bn5 = nn.BatchNorm3d(_c*2)

        self.tp_conv6 = nn.Conv3d(_c*4, _c, kernel_size=3, stride=1, padding=1, bias=True) # concat input from conv2, out:[16,120,120]
        self.bn6 = nn.BatchNorm3d(_c)

        self.tp_conv7 = nn.Conv3d(_c*2, 1, kernel_size=3, stride=1, padding=1, bias=True) # concat input from conv1, out:[32,240,240]
        
    def forward(self, h):

        h = self.tp_conv1(h)
        h1 = self.relu(self.bn1(h)) # out:[16,120,120,120]

        h = self.tp_conv2(h1)
        h2 = self.relu(self.bn2(h)) # out:[8,60,60,60]

        h = self.tp_conv3(h2)
        h = self.relu(self.bn3(h)) # out:[4,30,30,30]

        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h)) # out:[4,30,30,30]

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv5(h)
        h = self.relu(self.bn5(h)) # out:[8,60,60,60]

        h = torch.cat([h, h2], 1)

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h)) # out:[16,120,120,120]

        h = torch.cat([h, h1], 1)

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv7(h) # out:[32,240,240,240]

        h = torch.tanh(h)

        return h
