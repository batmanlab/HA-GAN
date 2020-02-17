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
from BRATS_dataset_slim import BRATS_dataset
from Model_GAN_240_SN_v3plus_pix2pix import Discriminator, Generator

import matplotlib.pyplot as plt

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

def main():
    BATCH_SIZE=4
    workers = 8

    _eps = 1e-15
    img_size = 240
    TOTAL_ITER = 200000
    log_step = 20
    lamb = 10

    #setting latent variable sizes
    latent_dim = 1000
    basename = str(img_size)+"_"+str(latent_dim)+"_l1_SN_v3plus_pix2pix"
    SAVE_MODEL = True

    g_iters = 1

    trainset = BRATS_dataset(img_size=img_size)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,drop_last=True,
                                           shuffle=False,num_workers=workers)

    gen_load = inf_train_gen(train_loader)

    G = Generator()
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
    loss_l1 = nn.L1Loss()

    fake_labels = torch.zeros((BATCH_SIZE, 1)).cuda()
    real_labels = torch.ones((BATCH_SIZE, 1)).cuda()

    summary_writer = SummaryWriter("./checkpoint/"+basename)

    for iteration in range(TOTAL_ITER):

    ###############################################
    # Train D
    ###############################################
        for p in D.parameters():  
            p.requires_grad = True 

        real_images_t1, real_images_flair = gen_load.__next__()
        D.zero_grad()
        real_images_t1 = Variable(real_images_t1).float().cuda()
        real_images_flair = Variable(real_images_flair).float().cuda()
        
        crop_idx = np.random.randint(0,129) # 160 - 32 + 1
        real_images_t1 = real_images_t1[:,:,crop_idx:crop_idx+32,:,:]
        real_images_flair = real_images_flair[:,:,crop_idx:crop_idx+32,:,:]

        real_images = torch.cat([real_images_t1, real_images_flair], 1)
        y_real_pred = D(real_images, crop_idx)

        d_real_loss = loss_f(y_real_pred, real_labels)
        
        fake_images_flair = G(real_images_t1)
        fake_images = torch.cat([real_images_t1, fake_images_flair.detach()], 1)
        
        y_fake_pred = D(fake_images, crop_idx)

        d_fake_loss = loss_f(y_fake_pred, fake_labels)
     
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()

        d_optimizer.step()

        ###############################################
        # Train G 
        ###############################################
        for p in D.parameters():
            p.requires_grad = False
            
        for iters in range(g_iters):
            G.zero_grad()
            
            real_images_t1, real_images_flair = gen_load.__next__()
            real_images_t1 = Variable(real_images_t1).float().cuda()
            real_images_flair = Variable(real_images_flair).float().cuda()

            crop_idx = np.random.randint(0,129) # 160 - 32 + 1
            real_images_t1 = real_images_t1[:,:,crop_idx:crop_idx+32,:,:]
            real_images_flair = real_images_flair[:,:,crop_idx:crop_idx+32,:,:]
        
            fake_images_flair = G(real_images_t1)
            fake_images = torch.cat([real_images_t1, fake_images_flair], 1)
            
            y_fake_g = D(fake_images, crop_idx)

            y_fake_loss = loss_f(y_fake_g, real_labels)
            y_l1_loss = loss_l1(fake_images_flair, real_images_flair)
            g_loss = y_fake_loss + lamb*y_l1_loss

            g_loss.backward()
            g_optimizer.step()

        # Logging
        if iteration%log_step == 0:
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
            summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
            summary_writer.add_scalar('G', y_fake_loss.item(), iteration)
            summary_writer.add_scalar('L1', y_l1_loss.item(), iteration)

        ###############################################
        # Visualization
        ###############################################
        if iteration%100 == 0:
            print('[{}/{}]'.format(iteration,TOTAL_ITER),
                  'D_real: {:<8.3}'.format(d_real_loss.item()),
                  'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
                  'G: {:<8.3}'.format(g_loss.item()),
                  'L1: {:<8.3}'.format(y_l1_loss.item()))

            featmask = np.squeeze((0.5*real_images_flair[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(120,120,16),figure=fig,draw_cross=False,cmap="bone")
            summary_writer.add_figure('Real', fig, iteration, close=True)
            
            featmask = np.squeeze((0.5*fake_images_flair[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(120,120,16),figure=fig,draw_cross=False,cmap="bone")
            summary_writer.add_figure('Fake', fig, iteration, close=True)
            
        if SAVE_MODEL and iteration > 5000 and (iteration+1)%500 == 0:
            torch.save(G.state_dict(),'./checkpoint/'+basename+'/G_iter'+str(iteration+1)+'.pth')
            torch.save(D.state_dict(),'./checkpoint/'+basename+'/D_iter'+str(iteration+1)+'.pth')
if __name__ == '__main__':
    main()
