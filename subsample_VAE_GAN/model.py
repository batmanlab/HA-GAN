import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable


from COPD_dataset_slim import COPD_dataset
import structure





class VAE_GAN(nn.Module):
    def inf_train_gen(self, data_loader):
        while True:
            for _,images in enumerate(data_loader):
                yield images
            
    def __init__(self, d_latent = 1024, batch_size = 4, workers = 8, 
                 log_step = 50, modelname = None, alpha = 100., beta = 1e-3, alpha_feature = None, 
                 feature = 3, hp_weight = 0, U_net = False, 
                 G_lr = 1e-4, D_lr = 4e-4, E_lr = 1e-4, g_iter = None, 
                 noise_std = .1, no_fc = False, feed_noise = False, 
                 channel = None, noise_channel = 256, 
                 img_size=256, dis_loss = "logistic", random_d = False):
        
        super(VAE_GAN, self).__init__()

        self.log_step = log_step
        
        self.alpha = alpha
        self.beta = beta
        
        self.feed_noise = feed_noise
        self.noise_channel = noise_channel
        self.alpha_feature = alpha_feature
        self.feature = feature
        self.hp_weight = hp_weight
        
        self.g_iter = g_iter
        self.no_fc = no_fc
        self.loss = dis_loss
        self.random_d = random_d
        self.U_net = U_net
        
        
        
        if modelname is None:
            self.modelname = \
            "_fea_{}_{}_feed_noise_{}_{}_chn{}_alpha{}_beta{}_lr{}_{}_{}_bs{}_gi{}"\
                .format(alpha_feature, feature, feed_noise, noise_channel, channel, alpha, beta, G_lr, D_lr, E_lr, batch_size, g_iter)
        else:
            self.modelname = modelname

        self.model_dir = self.modelname
            
        if not os.path.exists( self.model_dir ):
            os.mkdir( self.model_dir )
            os.mkdir( self.model_dir + "/image")
            os.mkdir( self.model_dir + "/model")

        self.log_file = open( self.model_dir + "/logfile.txt", "a", 1)
            
            
        trainset = COPD_dataset(img_size=img_size)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=workers)
        self.gen_load = self.inf_train_gen(train_loader)
        
        if self.feed_noise:
            if self.no_fc:
                self.G = nn.DataParallel( structure.Generator(  no_fc = no_fc, 
                                            channel = channel + noise_channel, U_net = self.U_net )
                                        ).cuda()
            else:
                self.G = nn.DataParallel( structure.Generator(  no_fc = no_fc, 
                                            d_latent = d_latent + noise_channel, U_net = self.U_net)
                                        ).cuda()
                
        else:
            self.G = nn.DataParallel( structure.Generator(
                    d_latent = d_latent, no_fc = no_fc, 
                    channel = channel, U_net = self.U_net ) ).cuda()
            
        self.D = nn.DataParallel( structure.Discriminator() ).cuda()
        
        if self.hp_weight!=0:
            self.E = nn.DataParallel( structure.Encoder(d_latent = d_latent, no_fc = no_fc, 
                               channel = channel, in_channel = 2 ) ).cuda()
        else:
            self.E = nn.DataParallel( structure.Encoder(d_latent = d_latent, no_fc = no_fc, 
                               channel = channel, in_channel = 1 ) ).cuda()
            
        self.sub_E = nn.DataParallel( 
            structure.Sub_Encoder(d_latent = d_latent, no_fc = no_fc, channel = channel  ) ).cuda()
        
        self.U_net_encode = nn.DataParallel( 
            structure.U_net_encode(64, 8) ).cuda()
        self.U_net_decode = nn.DataParallel( 
            structure.U_net_decode(8, 64) ).cuda()
        
        self.G_optim = optim.Adam(
            list( self.G.parameters() ) + list( self.U_net_decode.parameters() ),
            lr=G_lr, betas=(0.0,0.999), eps=1e-8 )
        self.D_optim = optim.Adam(
            self.D.parameters(), lr=D_lr, betas=(0.0,0.999), eps=1e-8)
        self.E_optim = optim.Adam(
            list( self.E.parameters() ) + list( self.sub_E.parameters() ) + \
            list( self.U_net_encode.parameters() ),
            lr=E_lr, betas=(0.0,0.999), eps=1e-8)

        self.D_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.recon_loss = nn.MSELoss(reduction='sum')
#         self.summary_writer = SummaryWriter( self.modelname + "/" + self.modelname)

        self.step = 0
    
        self.fake_labels = torch.zeros((batch_size, 1)).cuda()
        self.real_labels = torch.ones((batch_size, 1)).cuda()
        
        if self.hp_weight!=0:
        
            self.gf = self.build_Gaussian_filter(5, 1).cuda()

            kernel = torch.ones([1, 1, 5, 5, 5]).cuda()
            self.one_filter = torch.nn.Conv3d(in_channels=1, out_channels=1, padding = 2, 
                                        kernel_size=5, groups=1, bias=False).cuda()
            self.one_filter.weight.data = kernel
            self.one_filter.weight.requires_grad = False

    def hp_filter(self, X0):
        X = X0 - self.gf ( X0 )
        X = F.relu(X - .5)
        return X
    
    def KLD(self, mu, log_var):
        return .5 * ( -1 - log_var + log_var.exp() + mu.pow(2)  ).sum()
        

    def build_Gaussian_filter( self, kernel_size = 3, sigma = 1):

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)

        gaussian_kernel = torch.zeros([kernel_size, kernel_size, kernel_size])
        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        for iii in range(kernel_size):
            for jjj in range(kernel_size):
                for kkk in range(kernel_size):
                    gaussian_kernel[iii, jjj, kkk] = np.exp(
                              ( - (iii - mean) ** 2 - (jjj - mean) ** 2 - (kkk - mean) ** 2  ) / (2*variance)
                          )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

        gaussian_filter = torch.nn.Conv3d(in_channels=1, out_channels=1, padding = kernel_size // 2, 
                                    kernel_size=kernel_size, groups=1, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter        
    
    
    
    
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prod_gaussians(self, mu, log_var):
        
        var = torch.exp(log_var)
        var_res = 1. / ( 1. / var ).sum(0)
        log_var_res = torch.log( var_res )
        
        mu_res = ( mu / var ).sum(0) * var_res
        
        
        return mu_res, log_var_res
        
    def save_model(self):
        
        self.eval()
        torch.save(self.state_dict(), self.model_dir + "/model/model{}".format(self.step))


                
    def gen_loss(self, D_fake):
        return - D_fake.sum() 
        
    def dis_loss(self, D_fake, D_real):
        return - ( - torch.functional.F.relu( 1 - D_real).sum()
            - torch.functional.F.relu( 1 + D_fake).sum()
               ) 
        
            
    def train_VAE(self, steps = 100):
        for iii in range(steps):
            
            self.train()
            self.G.module.mode = "train"
            
            self.G_optim.zero_grad()
            self.E_optim.zero_grad()
            
            images = next(self.gen_load).float().cuda()
            images_small = F.interpolate( images, scale_factor = .25).cuda()
            
            crop_idx1 = np.random.randint(256 // 4 // 8 * 7) * 4 
            crop_idx2 = np.random.randint(256 // 4 // 8 * 7) * 4 
            
            crop_image1 = images[:, :, crop_idx1 : crop_idx1+32, :, :]
            crop_image2 = images[:, :, crop_idx2 : crop_idx2+32, :, :]
            
            
            mu1, log_var1 = self.E(crop_image1, crop_idx1)
            mu2, log_var2 = self.E(crop_image2, crop_idx2)
            
            mu_ = [ mu1.unsqueeze(0), mu2.unsqueeze(0) ]
            log_var_ = [ log_var1.unsqueeze(0), log_var2.unsqueeze(0) ]
            
            if self.step % 2 == 0:
                mu3, log_var3 = self.sub_E(images_small)
                mu_.append( mu3.unsqueeze(0) )
                log_var_.append( log_var3.unsqueeze(0) )
                
                
            mu, log_var = self.prod_gaussians( 
                torch.cat( mu_, 0) , 
                torch.cat( log_var_, 0)
            )
            
            z = self.reparameterize(mu, log_var)
            
            recon_crop1, recon_small = self.G(z, crop_idx1)
            recon_crop2, recon_small = self.G(z, crop_idx2)
            
            KLD = self.beta * self.KLD(mu, log_var)
            
            recon1 = self.recon_loss(recon_crop1, crop_image1)
            recon2 = self.recon_loss(recon_crop2, crop_image2)
            recon_small = self.recon_loss(recon_small, images_small) * 8.
            
            if iii % 2 == 0:
                loss =  recon1 + recon2 + KLD
            else:
                loss =  recon1 + recon2 + recon_small + KLD                
                
            
            
            loss.backward()
            
            txt = "Step{},{} Update G,E\n".format(self.step, gii) + \
                  "KLD:{}\n".format(KLD) + \
                  "recon1:{}\n".format(recon1) +\
                  "recon2:{}\n".format(recon2) +\
                  "recon small:{}\n".format(recon_small) +\
                  "loss:{}\n\n".format(loss)
            
            self.log_file.write(txt)
            print(txt)
            
            self.G_optim.step()
            self.E_optim.step()
            
            
            if self.step % self.log_step == 0:
                self.save_model()
            
            
            self.step += 1
        
    def train_GAN(self, steps = 100, warmup_step = 10):
        
        if self.g_iter is None:
            g_iter = 1
        else:
            g_iter = self.g_iter
        
        for iii in range(steps):
            self.train()
            
            if iii < warmup_step:
                self.warmup = True
                
#                 for mmm in self.E.module.parameters():
#                     mmm.requires_grad = False
                
#                 self.G.eval()
#                 self.E.eval()
            else:
                self.warmup = False

#                 for mmm in self.E.module.parameters():
#                     mmm.requires_grad = True
            
            


            ###############################################
            # Train G, E
            ###############################################                    
            for gii in range( g_iter ): 
                
                self.G_optim.zero_grad()
                self.E_optim.zero_grad()
                
                for p in self.D.parameters():  
                    p.requires_grad = False
                
                images = next(self.gen_load).float().cuda()
                images_small = F.interpolate( images, scale_factor = .25).cuda()

                batch_size = images.shape[0]

                crop_idx1 = np.random.randint(256 // 4 // 8 * 7) * 4 
                crop_idx2 = np.random.randint(256 // 4 // 8 * 7) * 4 

                
                
                crop_image1 = images[:, :, crop_idx1 : crop_idx1+32, :, :]
                crop_image2 = images[:, :, crop_idx2 : crop_idx2+32, :, :]
                if self.hp_weight > 0:        
                    with torch.no_grad():
                        image_fea1 = self.hp_filter(crop_image1)
                        image_fea2 = self.hp_filter(crop_image2)
                        
                    X1 = torch.cat([crop_image1, image_fea1], 1) 
                    X2 = torch.cat([crop_image2, image_fea2], 1) 
                else:
                    X1 = crop_image1
                    X2 = crop_image2

                if self.U_net:
                    mu1, log_var1, h_U1 = self.E( X1, crop_idx1, U_net = True )
                    mu2, log_var2, h_U2 = self.E( X2, crop_idx2, U_net = True )
                    
                    U_mu1, U_log_var1 = self.U_net_encode( h_U1 )
                    U_mu2, U_log_var2 = self.U_net_encode( h_U2 )
                    
                else:
                    mu1, log_var1 = self.E(X1, crop_idx1)
                    mu2, log_var2 = self.E(X2, crop_idx2)

                mu_ = [ mu1.unsqueeze(0), mu2.unsqueeze(0) ]
                log_var_ = [ log_var1.unsqueeze(0), log_var2.unsqueeze(0) ]
                


                if iii % 2 == 0:
                    mu3, log_var3 = self.sub_E(images_small)
                    mu_.append( mu3.unsqueeze(0) )
                    log_var_.append( log_var3.unsqueeze(0) )


                mu, log_var = self.prod_gaussians( 
                    torch.cat( mu_, 0) , 
                    torch.cat( log_var_, 0)
                )
                
                z = self.reparameterize(mu, log_var)
                
                if self.U_net:
                    U_mu_ = [ U_mu1.unsqueeze(0), U_mu2.unsqueeze(0) ]
                    U_log_var_ = [ U_log_var1.unsqueeze(0), U_log_var2.unsqueeze(0) ]
                    U_mu, U_log_var = self.prod_gaussians( 
                        torch.cat( U_mu_, 0) , 
                        torch.cat( U_log_var_, 0)
                    )
                    
                    U_z = self.reparameterize(U_mu, U_log_var)
                    
                    U_decoded =  self.U_net_decode(U_z)
                
                    


                
                if self.feed_noise:
                    z_size = list(z.shape)
                    z_size[1] = self.noise_channel
                    noise = torch.randn(z_size).cuda()
                    z = torch.cat( [z, noise], 1 )

                if self.U_net:
                    recon_crop1, recon_small = self.G( z, crop_idx1, U_decoded )
                    recon_crop2, recon_small = self.G( z, crop_idx2, U_decoded )
                else:
                    recon_crop1, recon_small = self.G( z, crop_idx1 )
                    recon_crop2, recon_small = self.G( z, crop_idx2 )


                KLD = self.beta * self.KLD(mu, log_var)
                if self.U_net:
                    KLD += self.beta * self.KLD( U_mu, U_log_var )

                recon1 = self.alpha * self.recon_loss(
                        recon_crop1, crop_image1) / (256 * 256 * 32)
                recon2 = self.alpha * self.recon_loss(
                        recon_crop2, crop_image2) / (256 * 256 * 32)
                recon_small_loss = self.alpha * self.recon_loss(
                        recon_small, images_small) / (64 * 64 * 64)
                
                if not self.alpha_feature is None:
                    recon_feature1 = self.D( recon_crop1, feature_extraction = self.feature )
                    recon_feature2 = self.D( recon_crop2, feature_extraction = self.feature )
                    image_feature1 = self.D( crop_image1, feature_extraction = self.feature )
                    image_feature2 = self.D( crop_image2, feature_extraction = self.feature )

                    n_voxel = np.prod( list( recon_feature1.shape ) )  

                    recon_feature1 = \
                        self.alpha_feature * self.recon_loss(recon_feature1, image_feature1) / n_voxel
                    recon_feature2 = \
                        self.alpha_feature * self.recon_loss(recon_feature2, image_feature2) / n_voxel
                else:
                    recon_feature1 = 0.
                    recon_feature2 = 0.
                    
                if self.hp_weight > 0:        
                    recon_fea1 = self.hp_filter(recon_crop1)
                    recon_fea2 = self.hp_filter(recon_crop2)
                    image_fea1 = self.hp_filter(crop_image1)
                    image_fea2 = self.hp_filter(crop_image2)

                    n_voxel = np.prod( list( recon_fea1.shape ) )  

                    recon_hp1 = \
                        self.hp_weight * self.recon_loss(recon_fea1, image_fea1 ) / n_voxel
                    recon_hp2 = \
                        self.hp_weight * self.recon_loss(recon_fea2, image_fea2 ) / n_voxel
                else:
                    recon_hp1 = 0.
                    recon_hp2 = 0.
                    
                y_fake_g1 = self.D(recon_crop1, recon_small, crop_idx1)

                y_fake_g2 = self.D(recon_crop2, recon_small, crop_idx2)

                if self.loss == "logistic": 
                    d_fake_loss1 = self.D_loss(y_fake_g1, self.real_labels)
                    d_fake_loss2 = self.D_loss(y_fake_g2, self.real_labels)

                    d_loss = ( d_fake_loss1 + d_fake_loss2 ) / 2

                else:
                    d_loss = ( self.gen_loss(y_fake_g1) + self.gen_loss(y_fake_g2) ) / 2

                    
                if self.warmup:
                    
                    if iii % 2 == 0:
                        loss =  recon1 + recon2 + \
                            recon_feature1 + recon_feature2 + recon_hp1 + recon_hp2 + KLD
                    else:
                        loss =  recon1 + recon2 + \
                            recon_feature1 + recon_feature2 + recon_hp1 + \
                            recon_small_loss + KLD 
                else:
                    if iii % 2 == 0:
                        loss =  recon1 + recon2 + \
                            recon_feature1 + recon_feature2 + recon_hp1 + recon_hp2 + KLD + d_loss
                    else:
                        loss =  recon1 + recon2 + \
                            recon_feature1 + recon_feature2 + recon_hp1 + \
                            recon_small_loss + KLD + d_loss
  
                loss = loss / batch_size

                loss.backward()

                txt = "Step{},{} Update G,E\n".format(self.step, gii) + \
                      "KLD:{}\n".format(KLD) + \
                      "recon1:{}\n".format(recon1) +\
                      "recon2:{}\n".format(recon2) +\
                      "recon_feature1:{}\n".format(recon_feature1) +\
                      "recon_feature2:{}\n".format(recon_feature2) +\
                      "recon_hp1:{}\n".format(recon_hp1) +\
                      "recon_hp2:{}\n".format(recon_hp2) +\
                      "recon small:{}\n".format(recon_small_loss) +\
                      "gen_loss:{}\n".format(d_loss) +\
                      "loss:{}\n\n".format(loss)
                
                if self.warmup:
                    txt += "Warming up\n"

                self.log_file.write(txt)
                print(txt)
            
                self.G_optim.step()
                
                self.E_optim.step()
            
            if self.g_iter is None:
                if d_loss < 10.:
                    g_iter = 1
                elif d_loss < 15.:
                    g_iter = 5
                elif d_loss < 20.:
                    g_iter = 10
                else:
                    g_iter = 20
                    
                if self.warmup:
                    g_iter = 1
                    
                txt = "g_iter:{}\n\n".format(g_iter)
                self.log_file.write(txt)
                print(txt)
                
            ###############################################
            # Train D
            ###############################################                    
            
            for p in self.D.parameters():  
                p.requires_grad = True
            
            self.D_optim.zero_grad()

            y_real_pred1 = self.D(crop_image1.detach(), images_small.detach(), crop_idx1)

            y_fake_pred1 = self.D(recon_crop1.detach(), recon_small.detach(), crop_idx1)
            
            
            if self.random_d and self.step % 5 == 0:
                y_fake_pred2 = self.D(crop_image2.detach(), images_small.detach(), crop_idx2)           
                y_real_pred2 = self.D(recon_crop2.detach(), images_small.detach(), crop_idx2)
            else:
                y_real_pred2 = self.D(crop_image2.detach(), images_small.detach(), crop_idx2)           
                y_fake_pred2 = self.D(recon_crop2.detach(), recon_small.detach(), crop_idx2)
            
            
            if self.loss == "logistic":             
                d_real_loss1 = self.D_loss(y_real_pred1, self.real_labels)
                d_real_loss2 = self.D_loss(y_real_pred2, self.real_labels)
                d_fake_loss1 = self.D_loss(y_fake_pred1, self.fake_labels)            
                d_fake_loss2 = self.D_loss(y_fake_pred2, self.fake_labels)            
                d_loss = d_real_loss1 + d_real_loss2 + d_fake_loss1 + d_fake_loss2
            else:
                d_loss = self.dis_loss(y_fake_pred1, y_real_pred1) + self.dis_loss(y_fake_pred2, y_real_pred2)
            
            d_loss = d_loss / batch_size
            
            d_loss.backward()

            txt = "Update D\n"+ \
                  "d_real1:{}\n".format( y_real_pred1.mean() ) +\
                  "d_real2:{}\n".format( y_real_pred2.mean() ) +\
                  "d_fake1:{}\n".format( y_fake_pred1.mean() ) +\
                  "d_fake2:{}\n".format( y_fake_pred2.mean() ) +\
                  "d_loss:{}\n\n".format(d_loss)

            self.log_file.write(txt)
            print(txt)
            
            
            self.D_optim.step()

            
            
            
            
            if self.step % self.log_step == 0:
                self.save_model()
            
            
            self.step += 1
            
    

    
    