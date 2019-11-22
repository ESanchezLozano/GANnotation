import glob, os, sys, csv, math, functools, collections, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from utils import LossNetwork, init_weights
from model_GAN import Generator, Discriminator
import itertools
from torch.autograd import Variable

def _gradient_penalty_D(D, real_img, fake_img, maps=None):
    alpha = torch.rand(real_img.size(0),1,1,1).cuda().expand_as(real_img)
    interpolated = alpha * real_img + (1 - alpha) * fake_img
    interpolated.required_grad = True
    if maps is not None:
        interpolated_prob,_ = D(torch.cat((interpolated,maps),1))
    else:
        interpolated_prob = D(interpolated)

    grad = torch.autograd.grad(outputs=interpolated_prob,
                               inputs=interpolated,
                               grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    grad = grad.view(grad.size(0),-1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    loss_d_gp = torch.mean((grad_l2norm-1)**2)
    return loss_d_gp

def TVLoss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = torch.numel(x[:,:,1:,:])
    count_w = torch.numel(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:] - x[:,:, :h_x -1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:, :w_x -1]),2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size




class GANnotation():

    def __init__(self, 
            losses = dict(adv=1, rec=100, self=100, triple=100, tv=1e-5, percep=0),
            gradclip=0, gantype='wgan-gp', edge=False
        ):
    
        self.npoints = 68
        self.gradclip = gradclip
        self.l = losses

        # - define models
        self.DIS = Discriminator(ndim=0)      
        self.GEN = Generator(conv_dim=64, c_dim=self.npoints if not edge else 3)
        init_weights(self.DIS)
        init_weights(self.GEN)
        if torch.cuda.device_count() > 1:
            self.GEN = torch.nn.DataParallel(self.GEN)
            self.DIS = torch.nn.DataParallel(self.DIS)
            
        self.DIS.to('cuda').train()
        self.GEN.to('cuda').train()

        if self.l['percep'] > 0:
            # - VGG for perceptual loss
            self.loss_network = LossNetwork(torch.nn.DataParallel(vgg16(pretrained=True))) if torch.cuda.device_count() > 1 else LossNetwork(vgg16(pretrained=True))
            self.loss_network.eval()
            self.loss_network.to('cuda')
            
        self.TripleLoss = (lambda x,y : torch.mean(torch.abs(x-y))) if self.l['triple'] > 0 else None
        self.SelfLoss = torch.nn.L1Loss().to('cuda') if self.l['self'] > 0 else None
        self.RecLoss = torch.nn.L1Loss().to('cuda') if self.l['rec'] > 0 else None
        self.PercepLoss = torch.nn.MSELoss().to('cuda') if self.l['percep'] > 0 else None
        self.TVLoss = TVLoss if self.l['tv'] > 0 else None
        self.gantype = gantype
        self.loss = dict(G=dict.fromkeys(['adv','self','triple','percep','rec','tv','all']), D=dict.fromkeys(['real','fake','gp','all']))

        
    def _advloss_D(self, score, real=True):
        if self.gantype == 'wgan-gp':
            return -torch.mean(score) if real else torch.mean(score)
        elif self.gantype == 'hinge':
            return torch.mean(torch.nn.ReLU()(1.0 - score)) if real else torch.mean(torch.nn.ReLU()(1.0 + score))

    def _pp_loss(self,fake_im,real_im):     
        vgg_fake = self.loss_network(fake_im)
        vgg_target = self.loss_network(real_im)
        perceptualLoss = 0
        for vgg_idx in range(0,4):
            perceptualLoss += self.PercepLoss(vgg_fake[vgg_idx], vgg_target[vgg_idx].detach())
        return perceptualLoss

    def _resume(self,path_fan, path_gen):
        self.DIS.load_state_dict(torch.load(path_dis))
        self.GEN.load_state_dict(torch.load(path_gen))
               
    def _save(self, path_to_models, epoch):
        torch.save(self.DIS.state_dict(), path_to_models + str(epoch) + '.dis.pth')
        torch.save(self.GEN.state_dict(), path_to_models + str(epoch) + '.gen.pth')

    def __call__(self, A, HB):
        return self.GEN(A,HB)

    def _forward_D(self, data):
        # - reset losses
        for p in self.DIS.parameters():
            p.requires_grad = True
        self.loss['D'].update( dict.fromkeys(self.loss['D'].keys(), None))
        self.DIS.zero_grad()
        # - generate fake image A -> B
        AB = self.GEN(data['A_Im'], data['B_Hm']).detach()
        AB.requires_grad = True
        fakeAB = self.DIS(AB)
        realAB = self.DIS(data['B_Im'])
        self.loss['D']['real'] = self._advloss_D(realAB, True)
        self.loss['D']['fake'] = self._advloss_D(fakeAB, False)
        self.loss['D']['gp'] = 10*_gradient_penalty_D(self.DIS, data['B_Im'], AB, maps=None) if self.gantype =='wgan-gp' else None
        self.loss['D']['all'] = sum(filter(lambda x : x is not None, self.loss['D'].values()))
        self.loss['D']['all'].backward()
        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.DIS.parameters(), 1, norm_type=2)
 
    def _forward_G(self, data):
        for p in self.DIS.parameters():
            p.requires_grad = False
        
        # - reset losses
        self.loss['G'].update( dict.fromkeys(self.loss['G'].keys(), None))
        self.GEN.zero_grad()

        # - adversarial loss
        AB = self.GEN(data['A_Im'], data['B_Hm'])
        pred_AB = self.DIS(AB)
        self.loss['G']['adv'] = -pred_AB.mean() * self.l['adv']

        # - reconstruction loss
        self.loss['G']['rec'] = self.l['rec'] * self.RecLoss( AB, data['B_Im']) if self.RecLoss is not None else None

        # - self-consistency loss
        if self.SelfLoss is not None:
            ABA = self.GEN(AB, data['A_Hm'])
            self.loss['G']['self'] = self.l['self'] * self.SelfLoss(ABA, data['A_Im'])

        # - triple consistency loss
        if self.TripleLoss is not None:
            AC = self.GEN(data['A_Im'], data['C_Hm'])
            BC = self.GEN(AB, data['C_Hm'])
            self.loss['G']['triple'] = self.l['triple'] * self.TripleLoss(AC, BC)

        # - perceptual loss
        self.loss['G']['percep'] = self.l['percep'] * self._pp_loss(AB, data['B_Im']) if self.PercepLoss is not None else None

        # - total variation loss
        self.loss['G']['tv'] = self.l['tv'] * self.TVLoss(AB) if self.TVLoss is not None else None

        # - all losses and backward
        self.loss['G']['all'] = sum(filter(lambda x : x is not None, self.loss['G'].values()))
        self.loss['G']['all'].backward()
        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.GEN.parameters(), 1, norm_type=2)
 
        return AB
    
