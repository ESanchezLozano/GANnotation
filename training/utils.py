import os, sys, time, torch, math, numpy as np, cv2, collections, itertools
import torch.nn as nn
import torch.nn.functional as F

################################### - classes 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.LossOutput = collections.namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        self.vgg_layers = vgg_model.features if hasattr(vgg_model,'features') else vgg_model.module.features #### to allow use in DataParallel
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.LossOutput(**output)


class EdgeMap(object):
    def __init__(self,out_res, num_parts=3):
        self.out_res = out_res
        self.num_parts = num_parts
        self.groups = [
            [np.arange(0,17,1), (255,0,0)],
            [np.arange(17,22,1), (0,255,0)],
            [np.arange(22,27,1), (0,255,0)],
            [np.arange(27,31,1), (0,0,255)],
            [np.arange(31,36,1), (0,0,255)],
            [list(np.arange(36,42,1))+[36], (0,255,0)],
            [list(np.arange(42,48,1))+[42], (0,255,0)],
            [list(np.arange(48,60,1))+[48], (0,0,255)],
            [list(np.arange(60,68,1))+[60], (0,0,255)]
        ]
        
    def __call__(self,shape):
        image = np.zeros((self.out_res, self.out_res, self.num_parts), dtype=np.float32)
        for g in self.groups:
            for i in range(len(g[0]) - 1):
                start = int(shape[g[0][i]][0]), int(shape[g[0][i]][1])
                end = int(shape[g[0][i+1]][0]), int(shape[g[0][i+1]][1])
                cv2.line(image, start, end, g[1], 3)
        
        return image.transpose(2,0,1)/255.0

class NumpyHeatmap(object):
    def __init__(self, out_res, num_parts=68):
        pass
    def __call__(self, shape):
        raise NameError('To be added')


################################ functions

def process_image(image,points,angle=0, flip=False, sigma=1,size=128, tight=16, hmsize=64):
    output = dict.fromkeys(['image','points','M'])
    if angle > 0:
        tmp_angle = np.clip(np.random.randn(1) * angle, -40.0, 40.0)
        image,points,M = affine_trans(image,points, tmp_angle)
        output['M'] = M
        tight = int(tight + 4*np.random.randn())
    image, points = crop( image , points, size, tight )
    if flip:
        image = cv2.flip(image, 1)
        groups = [(17,0,-1), (27,17,-1), (28,32,1), (36,31,-1), (46,42,-1), (48,46,-1),(40,36,-1),(42,40,-1),(55,48,-1),(60,55,-1),(65,60,-1),(68,65,-1)]
        pts_mirror = np.array(list(itertools.chain(*[range(k[0],k[1],k[2]) for k in groups]))) - 1
        points = (image.shape[1],0) + (-1,1)*points[pts_mirror,:]
            
    output['image'] = image
    output['points'] = np.floor(points)

    return output


def crop( image, landmarks , size, tight=8):
        delta_x = np.max(landmarks[:,0]) - np.min(landmarks[:,0])
        delta_y = np.max(landmarks[:,1]) - np.min(landmarks[:,1])
        delta = 0.5*(delta_x + delta_y)
        if delta < 20:
            tight_aux = 8
        else:
            tight_aux = int(tight * delta/100)
        pts_ = landmarks.copy()
        w = image.shape[1]
        h = image.shape[0]
        min_x = int(np.maximum( np.round( np.min(landmarks[:,0]) ) - tight_aux , 0 ))
        min_y = int(np.maximum( np.round( np.min(landmarks[:,1]) ) - tight_aux , 0 ))
        max_x = int(np.minimum( np.round( np.max(landmarks[:,0]) ) + tight_aux , w-1 ))
        max_y = int(np.minimum( np.round( np.max(landmarks[:,1]) ) + tight_aux , h-1 ))
        image = image[min_y:max_y,min_x:max_x,:]
        pts_[:,0] = pts_[:,0] - min_x
        pts_[:,1] = pts_[:,1] - min_y
        sw = size/image.shape[1]
        sh = size/image.shape[0]
        im = cv2.resize(image, dsize=(size,size),
                        interpolation=cv2.INTER_LINEAR)
        
        pts_[:,0] = pts_[:,0]*sw
        pts_[:,1] = pts_[:,1]*sh
        return im, pts_



def affine_trans(image,landmarks,angle=None,size=None):
    if angle is None:
        angle = 30*torch.randn(1)
       
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    dst = cv2.warpAffine(image, M, (nW,nH),borderMode=cv2.BORDER_REPLICATE)
    #print(landmarks.shape)
    new_landmarks = np.concatenate((landmarks,np.ones((landmarks.shape[0],1))),axis=1)
    if size is not None:
        sw = size/nW
        sh = size/nH
        dst = cv2.resize(dst, dsize=(size,size),
                        interpolation=cv2.INTER_LINEAR)
        M = [[sw,0],[0,sh]] @ M
    
    new_landmarks = new_landmarks.dot(M.transpose())
    return dst, new_landmarks, M



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



