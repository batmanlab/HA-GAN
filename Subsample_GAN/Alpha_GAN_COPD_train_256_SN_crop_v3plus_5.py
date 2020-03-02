import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import nibabel as nib
from nilearn import plotting

from utils import trim_state_dict_name, inf_train_gen
from COPD_dataset_slim import COPD_dataset
from Model_Alpha_GAN_256_SN_v3 import Discriminator, Generator, Encoder, Sub_Encoder

import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

def main():

    BATCH_SIZE=4
workers = 8

_eps = 1e-15
img_size = 256
TOTAL_ITER = 200000
log_step = 20
continue_step = 80500
threshold = -250

#setting latent variable sizes
latent_dim = 1024
basename = str(img_size)+"_"+str(latent_dim)+"_Alpha_SN_v3plus_5"
SAVE_MODEL = True

g_iters = 1

trainset = COPD_dataset(img_size=img_size, threshold=threshold)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,drop_last=True,
                                           shuffle=False,num_workers=workers)
gen_load = inf_train_gen(train_loader)

G = Generator(noise = latent_dim)
D = Discriminator()
E = Encoder()
Sub_E = Sub_Encoder(n_class = latent_dim)

if continue_step != 0:
    ckpt_path = './checkpoint/'+basename+'/G_iter'+str(continue_step)+'.pth'
    ckpt = torch.load(ckpt_path)
    ckpt = trim_state_dict_name(ckpt)
    G.load_state_dict(ckpt)
    ckpt_path = './checkpoint/'+basename+'/D_iter'+str(continue_step)+'.pth'
    ckpt = torch.load(ckpt_path)
    ckpt = trim_state_dict_name(ckpt)
    D.load_state_dict(ckpt)
    ckpt_path = './checkpoint/'+basename+'/E_iter'+str(continue_step)+'.pth'
    ckpt = torch.load(ckpt_path)
    ckpt = trim_state_dict_name(ckpt)
    E.load_state_dict(ckpt)
    ckpt_path = './checkpoint/'+basename+'/Sub_E_iter'+str(continue_step)+'.pth'
    ckpt = torch.load(ckpt_path)
    ckpt = trim_state_dict_name(ckpt)
    Sub_E.load_state_dict(ckpt)
    del ckpt
    print("Ckpt", continue_step, "loaded.")

G = nn.DataParallel(G).cuda()
D = nn.DataParallel(D).cuda()
E = nn.DataParallel(E).cuda()
Sub_E = nn.DataParallel(Sub_E).cuda()

G.train()
D.train()
E.train()
Sub_E.train()

g_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0,0.999), eps=1e-8)
d_optimizer = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0,0.999), eps=1e-8)
e_optimizer = optim.Adam(E.parameters(), lr=0.0001, betas=(0.0,0.999), eps=1e-8)
sub_e_optimizer = optim.Adam(Sub_E.parameters(), lr=0.0001, betas=(0.0,0.999), eps=1e-8)

real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())

loss_f = nn.BCEWithLogitsLoss()
loss_mse = nn.MSELoss()

fake_labels = torch.zeros((BATCH_SIZE, 1)).cuda()
real_labels = torch.ones((BATCH_SIZE, 1)).cuda()

summary_writer = SummaryWriter("./checkpoint/"+basename)

for p in D.parameters():  
    p.requires_grad = False
for p in G.parameters():  
    p.requires_grad = False
for p in E.parameters():  
    p.requires_grad = False
for p in Sub_E.parameters():  
    p.requires_grad = False

for iteration in range(continue_step, TOTAL_ITER):

    ###############################################
    # Train D
    ###############################################
    for p in D.parameters():  
        p.requires_grad = True
    for p in Sub_E.parameters():  
        p.requires_grad = False

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
    y_fake_pred = D(fake_images, fake_images_small, crop_idx)
    d_fake_loss = loss_f(y_fake_pred, fake_labels)
 
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()

    d_optimizer.step()

    ###############################################
    # Train G
    ###############################################
    for p in D.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = True
        
    for iters in range(g_iters):
        G.zero_grad()
        
        noise = torch.randn((BATCH_SIZE, latent_dim, 1, 1, 1)).cuda()
        fake_images, fake_images_small, fake_latent = G(noise, crop_idx, return_latent=True)
        fake_latent_crop = fake_latent[:,:,crop_idx//4:crop_idx//4+8,:,:]
        
        y_fake_g = D(fake_images, fake_images_small, crop_idx)
        g_loss = loss_f(y_fake_g, real_labels)
        g_loss.backward()
        g_optimizer.step()
                      
    ###############################################
    # Train E
    ###############################################
    for p in E.parameters():
        p.requires_grad = True
    for p in G.parameters():
        p.requires_grad = False
    E.zero_grad()
    
    z_hat = E(real_images_crop)
    x_hat = G(z_hat, crop_idx=None)
    
    e_loss = loss_mse(x_hat,real_images_crop)
    e_loss.backward()
    e_optimizer.step()

    ###############################################
    # Train Sub E
    ###############################################
    for p in Sub_E.parameters():
        p.requires_grad = True
    for p in E.parameters():
        p.requires_grad = False
    Sub_E.zero_grad()
    
    with torch.no_grad():
        z_hat_i_list = []
        for crop_idx_i in range(0,256,32):
            real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+32,:,:]
            z_hat_i = E(real_images_crop_i)
            z_hat_i_list.append(z_hat_i)
        z_hat = torch.cat(z_hat_i_list, dim=2).detach()   
    sub_z_hat = Sub_E(z_hat)
    sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx)
    
    sub_e_loss = (loss_mse(sub_x_hat_rec,real_images_crop) + loss_mse(sub_x_hat_rec_small,real_images_small))/2.

    sub_e_loss.backward()
    sub_e_optimizer.step()
    
    # Logging
    if iteration%log_step == 0:
        summary_writer.add_scalar('D', d_loss.item(), iteration)
        summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
        summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
        summary_writer.add_scalar('G_fake', g_loss.item(), iteration)
        summary_writer.add_scalar('E', e_loss.item(), iteration)
        summary_writer.add_scalar('Sub_E', sub_e_loss.item(), iteration)

    ###############################################
    # Visualization
    ###############################################
    if iteration%100 == 0:
        print('[{}/{}]'.format(iteration,TOTAL_ITER),
              'D_real: {:<8.3}'.format(d_real_loss.item()),
              'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
              'G_fake: {:<8.3}'.format(g_loss.item()),
              'Sub_E: {:<8.3}'.format(sub_e_loss.item()),
              'E: {:<8.3}'.format(e_loss.item()))

        featmask = np.squeeze((0.5*real_images_crop[0]+0.5).data.cpu().numpy())
        featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
        fig=plt.figure()
        plotting.plot_img(featmask,title="REAL",cut_coords=(128,128,16),figure=fig,draw_cross=False,cmap="bone")
        summary_writer.add_figure('Real', fig, iteration, close=True)

        featmask = np.squeeze((0.5*sub_x_hat_rec[0]+0.5).data.cpu().numpy())
        featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
        fig=plt.figure()
        plotting.plot_img(featmask,title="REC",cut_coords=(128,128,16),figure=fig,draw_cross=False,cmap="bone")
        summary_writer.add_figure('Rec', fig, iteration, close=True)
        
        featmask = np.squeeze((0.5*fake_images[0]+0.5).data.cpu().numpy())
        featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
        fig=plt.figure()
        plotting.plot_img(featmask,title="FAKE",cut_coords=(128,128,16),figure=fig,draw_cross=False,cmap="bone")
        summary_writer.add_figure('Fake', fig, iteration, close=True)
        
    if SAVE_MODEL and iteration > 20000 and (iteration+1)%500 == 0:
        torch.save(G.state_dict(),'./checkpoint/'+basename+'/G_iter'+str(iteration+1)+'.pth')
        torch.save(D.state_dict(),'./checkpoint/'+basename+'/D_iter'+str(iteration+1)+'.pth')
        torch.save(E.state_dict(),'./checkpoint/'+basename+'/E_iter'+str(iteration+1)+'.pth')
        torch.save(Sub_E.state_dict(),'./checkpoint/'+basename+'/Sub_E_iter'+str(iteration+1)+'.pth')

if __name__ == '__main__':
    main()