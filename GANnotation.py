import cv2
import torch
import numpy as np
import scipy.io as sio
import os
from model import Generator
import utils

class GANnotation:

    def __init__(self, path_to_model='',enable_cuda=True, train=False):
        self.GEN = Generator()
        self.enable_cuda = enable_cuda
        self.size = 128
        self.tight = 16
        net_weigths = torch.load(path_to_model,map_location=lambda storage, loc: storage)
        net_dict = {k.replace('module.',''): v for k, v in net_weigths['state_dict'].items()}
        self.GEN.load_state_dict(net_dict)
        if self.enable_cuda:
            self.GEN = self.GEN.cuda()
        self.GEN.eval()

        
    def reenactment(self,image,videocoords):
        #image, points = utils.process_image(image,coords,angle=0, flip=False, sigma=1,size=128, tight=16) # do this outside
        frame_w = int(np.floor(2*videocoords.max()))
        frame = np.zeros((frame_w,frame_w,3))
        if videocoords.ndim == 2:
            videocoords = videocoords.reshape((66,2,1))
        n_frames = videocoords.shape[2]
        cropped_points = np.zeros((66,2,n_frames))
        images = []
        for i in range(0,n_frames):
            print(i)
            if videocoords[0,0,i] > 0:
                target = videocoords[:,:,i]
                _, target = utils.crop( frame , target, size=128, tight=16 )
                cropped_points[:,:,i] = target
                A_to_B = utils.generate_Ginput(image,target,sigma=1,size=128).unsqueeze(0)
                if self.enable_cuda:
                    A_to_B = A_to_B.cuda()
                generated = 0.5*(self.GEN(torch.autograd.Variable(A_to_B)).data[0,:,:,:].cpu().numpy().swapaxes(0,1).swapaxes(1,2) + 1)
                imout = (255*generated).astype('uint8')
                images.append(imout)
        return images, cropped_points


 
    











