import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import nibabel as nib

from nilearn import plotting
from COPD_dataset_slim import COPD_dataset
from Model_GAN_256_SN_v6 import Discriminator, Generator

import matplotlib.pyplot as plt

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

def main():
    BATCH_SIZE=4
    workers = 8

    _eps = 1e-15
    img_size = 256
    TOTAL_ITER = 200000
    log_step = 20

    #setting latent variable sizes
    latent_dim = 1000
    basename = str(img_size)+"_"+str(latent_dim)+"_SN_v6"
    SAVE_MODEL = True

    g_iters = 1

    trainset = COPD_dataset(img_size=img_size)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,drop_last=True,
                                               shuffle=False,num_workers=workers)

    gen_load = inf_train_gen(train_loader)

    G = Generator(noise = latent_dim)
    D = Discriminator()

    G = nn.DataParallel(G).cuda()
    D = nn.DataParallel(D).cuda()

    G.train()
    D.train()

    g_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0,0.999), eps=1e-8)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0,0.999), eps=1e-8)

    real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
    fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
    loss_f = nn.BCEWithLogitsLoss()

    fake_labels = torch.zeros((BATCH_SIZE, 1)).cuda()
    real_labels = torch.ones((BATCH_SIZE, 1)).cuda()

    summary_writer = SummaryWriter("./checkpoint/"+basename)

    for iteration in range(TOTAL_ITER):

        ###############################################
        # Train D 
        ###############################################
        for p in D.parameters():  
            p.requires_grad = True 

        real_images = gen_load.__next__()
        D.zero_grad()
        real_images = Variable(real_images).float().cuda()
        real_images_small = F.interpolate(real_images, scale_factor = 0.25)
        
        crop_idx = np.random.randint(0,225) # 256 * 7/8 + 1
        real_images_crop = real_images[:,:,crop_idx:crop_idx+32,:,:]

        y_real_pred = D(real_images_crop, real_images_small, crop_idx)

        d_real_loss = loss_f(y_real_pred, real_labels)
        
        noise = torch.randn((BATCH_SIZE, latent_dim, 1, 1, 1)).cuda()
        fake_images, fake_images_small = G(noise, crop_idx)
        y_fake_pred = D(fake_images.detach(), fake_images_small.detach(), crop_idx)

        d_fake_loss = loss_f(y_fake_pred, fake_labels)
     
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        d_optimizer.step()

        ###############################################
        # Train G 
        ###############################################
        for p in D.parameters():
            p.requires_grad = False
            
        for iters in range(g_iters):
            G.zero_grad()
            noise = torch.randn((BATCH_SIZE, latent_dim, 1, 1, 1)).cuda()
            fake_images, fake_images_small = G(noise, crop_idx)
            y_fake_g = D(fake_images, fake_images_small, crop_idx)

            g_loss = loss_f(y_fake_g, real_labels)

            g_loss.backward()
            g_optimizer.step()

        # Logging
        if iteration%log_step == 0:
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
            summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
            summary_writer.add_scalar('G', g_loss.item(), iteration)

        ###############################################
        # Visualization
        ###############################################
        if iteration%100 == 0:
            print('[{}/{}]'.format(iteration,TOTAL_ITER),
                  'D: {:<8.3}'.format(d_loss.item()), 
                  'D_real: {:<8.3}'.format(d_real_loss.item()),
                  'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
                  'G: {:<8.3}'.format(g_loss.item()))

            featmask = np.squeeze((0.5*real_images_crop[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(128,128,16),figure=fig,draw_cross=False,cmap="bone")
            summary_writer.add_figure('Real', fig, iteration, close=True)
            
            featmask = np.squeeze((0.5*fake_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(128,128,16),figure=fig,draw_cross=False,cmap="bone")
            summary_writer.add_figure('Fake', fig, iteration, close=True)
            
        if SAVE_MODEL and iteration > 20000 and (iteration+1)%500 == 0:
            torch.save(G.state_dict(),'./checkpoint/'+basename+'/G_iter'+str(iteration+1)+'.pth')
            torch.save(D.state_dict(),'./checkpoint/'+basename+'/D_iter'+str(iteration+1)+'.pth')
            
if __name__ == '__main__':
    main()