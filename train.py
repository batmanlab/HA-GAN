#!/usr/bin/env python

# train HA-GAN
# Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis
# https://ieeexplore.ieee.org/abstract/document/9770375

import numpy as np
import torch
import os
import json
import argparse

from torch import nn
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import nibabel as nib
from nilearn import plotting

from utils import trim_state_dict_name, inf_train_gen
from volume_dataset import Volume_Dataset

import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch HA-GAN Training')
parser.add_argument('--batch-size', default=4, type=int,
                    help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--img-size', default=256, type=int,
                    help='size of training images (default: 256, can be 128 or 256)')
parser.add_argument('--num-iter', default=80000, type=int,
                    help='number of iteration for training (default: 80000)')
parser.add_argument('--log-iter', default=20, type=int,
                    help='number of iteration between logging (default: 20)')
parser.add_argument('--continue-iter', default=0, type=int,
                    help='continue from a ckeckpoint that has run for n iteration  (0 if a new run)')
parser.add_argument('--latent-dim', default=1024, type=int,
                    help='size of the input latent variable')
parser.add_argument('--g-iter', default=1, type=int,
                    help='number of generator pass per iteration')
parser.add_argument('--lr-g', default=0.0001, type=float,
                    help='learning rate for the generator')
parser.add_argument('--lr-d', default=0.0004, type=float,
                    help='learning rate for the discriminator')
parser.add_argument('--lr-e', default=0.0001, type=float,
                    help='learning rate for the encoder')
parser.add_argument('--data-dir', type=str,
                    help='path to the preprocessed data folder')
parser.add_argument('--exp-name', default='HA_GAN_run1', type=str,
                    help='name of the experiment')
parser.add_argument('--fold', default=0, type=int,
                    help='fold number for cross validation')

# configs for conditional generation
parser.add_argument('--lambda-class', default=0.1, type=float,
                    help='weights for the auxiliary classifier loss')
parser.add_argument('--num-class', default=0, type=int,
                    help='number of class for auxiliary classifier (0 if unconditional)')

def main():
    # Configuration
    args = parser.parse_args()

    trainset = Volume_Dataset(data_dir=args.data_dir, fold=args.fold, num_class=args.num_class)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,drop_last=True,
                                               shuffle=False,num_workers=args.workers)
    gen_load = inf_train_gen(train_loader)
    
    if args.img_size == 256:
        from models.Model_HA_GAN_256 import Discriminator, Generator, Encoder, Sub_Encoder
    elif args.img_size == 128:
        from models.Model_HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder
    else:
        raise NotImplmentedError
        
    G = Generator(mode='train', latent_dim=args.latent_dim, num_class=args.num_class).cuda()
    D = Discriminator(num_class=args.num_class).cuda()
    E = Encoder().cuda()
    Sub_E = Sub_Encoder(latent_dim=args.latent_dim).cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0,0.999), eps=1e-8)
    d_optimizer = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0,0.999), eps=1e-8)
    e_optimizer = optim.Adam(E.parameters(), lr=args.lr_e, betas=(0.0,0.999), eps=1e-8)
    sub_e_optimizer = optim.Adam(Sub_E.parameters(), lr=args.lr_e, betas=(0.0,0.999), eps=1e-8)

    # Resume from a previous checkpoint
    if args.continue_iter != 0:
        ckpt_path = './checkpoint/'+args.exp_name+'/G_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
        g_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/D_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/E_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        E.load_state_dict(ckpt['model'])
        e_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/Sub_E_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        Sub_E.load_state_dict(ckpt['model'])
        sub_e_optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        print("Ckpt", args.exp_name, args.continue_iter, "loaded.")

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    E = nn.DataParallel(E)
    Sub_E = nn.DataParallel(Sub_E)

    G.train()
    D.train()
    E.train()
    Sub_E.train()

    real_y = torch.ones((args.batch_size, 1)).cuda()
    fake_y = torch.zeros((args.batch_size, 1)).cuda()

    loss_f = nn.BCEWithLogitsLoss()
    loss_mse = nn.L1Loss()

    fake_labels = torch.zeros((args.batch_size, 1)).cuda()
    real_labels = torch.ones((args.batch_size, 1)).cuda()

    summary_writer = SummaryWriter("./checkpoint/"+args.exp_name)
    
    # save configurations to a dictionary
    with open(os.path.join("./checkpoint/"+args.exp_name, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    for p in D.parameters():  
        p.requires_grad = False
    for p in G.parameters():  
        p.requires_grad = False
    for p in E.parameters():  
        p.requires_grad = False
    for p in Sub_E.parameters():  
        p.requires_grad = False

    for iteration in range(args.continue_iter, args.num_iter):

        ###############################################
        # Train Discriminator (D^H and D^L)
        ###############################################
        for p in D.parameters():  
            p.requires_grad = True
        for p in Sub_E.parameters():  
            p.requires_grad = False

        real_images, class_label = gen_load.__next__()
        D.zero_grad()
        real_images = real_images.float().cuda()
        # low-res full volume of real image
        real_images_small = F.interpolate(real_images, scale_factor = 0.25)
        
        # randomly select a high-res sub-volume from real image
        crop_idx = np.random.randint(0,args.img_size*7/8+1) # 256 * 7/8 + 1
        real_images_crop = real_images[:,:,crop_idx:crop_idx+args.img_size//8,:,:]

        if args.num_class == 0: # unconditional
            y_real_pred = D(real_images_crop, real_images_small, crop_idx)
            d_real_loss = loss_f(y_real_pred, real_labels)

            # random generation
            noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
            # fake_images: high-res sub-volume of generated image
            # fake_images_small: low-res full volume of generated image
            fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=None)
            y_fake_pred = D(fake_images, fake_images_small, crop_idx)

        else: # conditional
            class_label_onehot = F.one_hot(class_label, num_classes=args.num_class)
            class_label = class_label.long().cuda()
            class_label_onehot = class_label_onehot.float().cuda()

            y_real_pred, y_real_class = D(real_images_crop, real_images_small, crop_idx)
            # GAN loss + auxiliary classifier loss
            d_real_loss = loss_f(y_real_pred, real_labels) + \
                          F.cross_entropy(y_real_class, class_label)

            # random generation
            noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
            fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=class_label_onehot)
            y_fake_pred, y_fake_class= D(fake_images, fake_images_small, crop_idx)

        d_fake_loss = loss_f(y_fake_pred, fake_labels)
     
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        d_optimizer.step()

        ###############################################
        # Train Generator (G^A, G^H and G^L)
        ###############################################
        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True
            
        for iters in range(args.g_iter):
            G.zero_grad()
            
            noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
            if args.num_class == 0: # unconditional
                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=None)

                y_fake_g = D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels)
            else: # conditional
                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=class_label_onehot)

                y_fake_g, y_fake_g_class = D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels) + \
                         args.lambda_class * F.cross_entropy(y_fake_g_class, class_label)

            g_loss.backward()
            g_optimizer.step()
                          
        ###############################################
        # Train Encoder (E^H)
        ###############################################
        for p in E.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False
        E.zero_grad()
        
        z_hat = E(real_images_crop)
        x_hat = G(z_hat, crop_idx=None)
        
        e_loss = loss_mse(x_hat, real_images_crop)
        e_loss.backward()
        e_optimizer.step()

        ###############################################
        # Train Sub Encoder (E^G)
        ###############################################
        for p in Sub_E.parameters():
            p.requires_grad = True
        for p in E.parameters():
            p.requires_grad = False
        Sub_E.zero_grad()
        
        with torch.no_grad():
            z_hat_i_list = []
            # Process all sub-volume and concatenate
            for crop_idx_i in range(0,args.img_size,args.img_size//8):
                real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+args.img_size//8,:,:]
                z_hat_i = E(real_images_crop_i)
                z_hat_i_list.append(z_hat_i)
            z_hat = torch.cat(z_hat_i_list, dim=2).detach()   
        sub_z_hat = Sub_E(z_hat)
        # Reconstruction
        if args.num_class == 0: # unconditional
            sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx)
        else: # conditional
            sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx, class_label=class_label_onehot)
        
        sub_e_loss = (loss_mse(sub_x_hat_rec,real_images_crop) + loss_mse(sub_x_hat_rec_small,real_images_small))/2.

        sub_e_loss.backward()
        sub_e_optimizer.step()
        
        # Logging
        if iteration%args.log_iter == 0:
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
            summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
            summary_writer.add_scalar('G_fake', g_loss.item(), iteration)
            summary_writer.add_scalar('E', e_loss.item(), iteration)
            summary_writer.add_scalar('Sub_E', sub_e_loss.item(), iteration)

        ###############################################
        # Visualization with Tensorboard
        ###############################################
        if iteration%200 == 0:
            print('[{}/{}]'.format(iteration,args.num_iter),
                  'D_real: {:<8.3}'.format(d_real_loss.item()),
                  'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
                  'G_fake: {:<8.3}'.format(g_loss.item()),
                  'Sub_E: {:<8.3}'.format(sub_e_loss.item()),
                  'E: {:<8.3}'.format(e_loss.item()))

            featmask = np.squeeze((0.5*real_images_crop[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Real', fig, iteration, close=True)

            featmask = np.squeeze((0.5*sub_x_hat_rec[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REC",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Rec', fig, iteration, close=True)
            
            featmask = np.squeeze((0.5*fake_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Fake', fig, iteration, close=True)
            
        if iteration > 30000 and (iteration+1)%500 == 0:
            torch.save({'model':G.state_dict(), 'optimizer':g_optimizer.state_dict()},'./checkpoint/'+args.exp_name+'/G_iter'+str(iteration+1)+'.pth')
            torch.save({'model':D.state_dict(), 'optimizer':d_optimizer.state_dict()},'./checkpoint/'+args.exp_name+'/D_iter'+str(iteration+1)+'.pth')
            torch.save({'model':E.state_dict(), 'optimizer':e_optimizer.state_dict()},'./checkpoint/'+args.exp_name+'/E_iter'+str(iteration+1)+'.pth')
            torch.save({'model':Sub_E.state_dict(), 'optimizer':sub_e_optimizer.state_dict()},'./checkpoint/'+args.exp_name+'/Sub_E_iter'+str(iteration+1)+'.pth')

if __name__ == '__main__':
    main()
