import model
import torch



self = model.VAE_GAN(batch_size = 2, d_latent = 512, alpha = 0., beta = 1e-10, 
                    G_lr = 1e-4, D_lr = 4e-4, E_lr = 1e-4, g_iter = 1, no_fc = False,
                    alpha_feature = 1., random_d = True, 
                    feature = 3, hp_weight = 0, U_net = False, 
                    channel = 1024, dis_loss = "hinge", feed_noise = True, noise_channel = 512 )


self.train_GAN(100000, 0)
