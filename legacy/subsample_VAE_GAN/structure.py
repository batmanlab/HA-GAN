import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from layers import SNConv3d, SNLinear



class Sub_Encoder(nn.Module):
    def __init__(self, channel = None, d_latent = 1024, no_fc = False):
        super(Sub_Encoder, self).__init__()
        
        if channel is None:
            channel = 256
            
        self.channel = channel
        self.d_latent = d_latent
        self.no_fc = no_fc

        self.conv1 = SNConv3d(1, channel//8, kernel_size=3, stride=2, padding=1) # in:[64,64,64], out:[32,32,32]
        self.bn1 = nn.BatchNorm3d( channel//8 )
        
        self.conv2 = SNConv3d(channel//8, channel//4, kernel_size=3, stride=2, padding=1) # out:[16,16,16]
        self.bn2 = nn.BatchNorm3d( channel//4 )
        
        self.conv3 = SNConv3d(channel//4, channel//2, kernel_size=3, stride=2, padding=1) # out:[8,8,8]
        self.bn3 = nn.BatchNorm3d( channel//2 )
        
        if no_fc:
            self.conv4 = SNConv3d(channel//2, channel * 2, kernel_size=3, stride=2, padding=1) # out:[4,4,4]
        else:
        
            self.conv4 = SNConv3d(channel//2, channel, kernel_size=3, stride=2, padding=1) # out:[4,4,4]
            self.bn4 = nn.BatchNorm3d( channel )

            self.conv5 = SNConv3d(channel, d_latent * 2, kernel_size=4, stride=1, padding=0) # out:[1,1,1,1]
        
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        if self.no_fc:
            h = self.conv4(h)
            return h[:, :self.channel, :, :, :], h[:, self.channel:, :, :, :] # mu, log_var
        
        else:
            h = F.relu(self.bn4(self.conv4(h)))
            h = self.conv5(h).view( -1,self.d_latent * 2 )
        
            return h[:, :self.d_latent], h[:, self.d_latent:] # mu, log_var
    

class Encoder(nn.Module):
    def __init__(self, channel=None, d_latent = 1024, no_fc = False, in_channel = 1):
        super(Encoder, self).__init__()        
        if channel is None:
            channel = 512
        self.channel = channel
        self.d_latent = d_latent
        self.no_fc = no_fc
        
        self.conv1 = SNConv3d(in_channel, channel//32, kernel_size=3, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
        self.bn1 = nn.BatchNorm3d( channel//32 )        
        
        self.conv2 = SNConv3d(channel//32, channel//16, kernel_size=3, stride=2, padding=1) # out:[8,64,64]
        self.bn2 = nn.BatchNorm3d( channel//16 )
        
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=3, stride=2, padding=1) # in:[64, 64, 64] out:[32,32,32]
        self.bn3 = nn.BatchNorm3d( channel//8 )

        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=3, stride=2, padding=1) # out:[16,16,16]
        self.bn4 = nn.BatchNorm3d( channel//4 )

        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=3, stride=2, padding=1) # out:[8,8,8]
        self.bn5 = nn.BatchNorm3d( channel//2 )

        
        if self.no_fc:
            self.conv6 = SNConv3d(channel//2, channel * 2, kernel_size=3, stride=2, padding=1) # out:[4,4,4]
        else:
            self.conv6 = SNConv3d(channel//2, channel, kernel_size=3, stride=2, padding=1) # out:[4,4,4]
            self.bn6 = nn.BatchNorm3d( channel )
            self.conv7 = SNConv3d(channel, channel//4, kernel_size= 4, stride=1, padding=0) # out:[1, 1, 1]

            self.fc = SNLinear(channel//4, d_latent * 2)
        
    def forward(self, x, crop_idx, U_net = False):
        batch_size = x.shape[0]
        

        
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        
        
        h0 = torch.zeros( [ batch_size, self.channel//16, 64, 64, 64] ).float().cuda()
        h0[:,:,crop_idx//4:crop_idx//4+8,:,:] = h
        
        
        
        h = F.relu(self.bn3(self.conv3(h0)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        
        if self.no_fc:
            h = self.conv6(h)
            if U_net:
                return h[:, :self.channel, :, :, :],  h[:, self.channel:, :, :, :], h0
            else:
                return h[:, :self.channel, :, :, :],  h[:, self.channel:, :, :, :]
        
        
        else:
            h = F.relu(self.bn6(self.conv6(h)))
            h = F.relu(self.conv7(h)).squeeze()

            h = self.fc(h).view(-1, self.d_latent * 2)
            return h[:, :self.d_latent], h[:, self.d_latent:]
            
    
class U_net_encode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(U_net_encode, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = SNConv3d(in_channel, in_channel // 2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d( in_channel//2 )        
        self.conv2 = SNConv3d(in_channel // 2, in_channel // 4, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d( in_channel//4 )
        self.conv3 = SNConv3d(in_channel // 4, out_channel * 2, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.conv3(h)
        return h[:, :self.out_channel], h[:, self.out_channel:]
        
class U_net_decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(U_net_decode, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = SNConv3d(in_channel, in_channel * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d( in_channel * 2 )        
        self.conv2 = SNConv3d(in_channel * 2, in_channel * 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d( in_channel * 4 )
        self.conv3 = SNConv3d(in_channel * 4, out_channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        h = F.interpolate(x,scale_factor = 2)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.interpolate(h,scale_factor = 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.interpolate(h,scale_factor = 2)
        h = self.conv3(h)
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
        
#         self.conv1 = SNConv3d(1, channel//32, kernel_size=4, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
#         self.conv2 = SNConv3d(channel//32, channel//16, kernel_size=4, stride=2, padding=1) # out:[8,64,64,64]
#         self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[4,32,32,32]
#         self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[2,16,16,16]
#         self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[1,8,8,8]
#         self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,4,4,4]
#         self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,4,4), stride=1, padding=0) # out:[1,1,1,1]
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

    def forward(self, h, h_small = None, crop_idx = 0, feature_extraction = None):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        if feature_extraction == 1:
            return h
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        if feature_extraction == 2:
            return h
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        if feature_extraction == 3:
            return h
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        if feature_extraction == 4:
            return h
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        if feature_extraction == 5:
            return h
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze(2).squeeze(2).squeeze(2)
        h = torch.cat([h, (crop_idx / 224. * torch.ones((h.size(0), 1))).cuda()], 1) # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h = self.fc2(h)
        
        if h_small is None:
            return h
        else:
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
    def __init__(self, d_latent:int = 1024 , channel=None, mode="train", no_fc = False, U_net = False, U_net_channel = 64):
        super(Generator, self).__init__()
        if channel is None:
            channel = 64
        else:
            channel = channel // 16
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.d_latent = d_latent
        self.no_fc = no_fc
        self.U_net = U_net
        
        if not no_fc:
            self.tp_conv1 = nn.ConvTranspose3d(d_latent, _c*16, 
                                           kernel_size=4, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm3d(_c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(_c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(_c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(_c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(_c*2)
            
        if self.U_net:
            self.tp_conv6 = nn.Conv3d(_c*2 + U_net_channel, _c, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(_c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
        if self.U_net:
            self.sub_G = Sub_Generator(channel = channel // 2 + U_net_channel // 4)
        else:
            self.sub_G = Sub_Generator(channel = channel // 2)
 
    def forward(self, z, crop_idx, U_net_input = None):
        if self.no_fc:
            h = z
        else:
            z = z.view(-1, self.d_latent ,1,1,1)
            h = self.tp_conv1(z)
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
        
        if self.U_net:
            h = torch.cat([h, U_net_input], 1)

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