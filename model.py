import torch
import torch.nn as nn
import torch.nn.init as weight_init
import csv
import math
import torch.nn.functional as F
import functools

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                weight_init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                weight_init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                weight_init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                weight_init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                weight_init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            weight_init.normal_(m.weight.data, 1.0, gain)
            weight_init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def downsample_block(input_channels=64, output_channels=128, norm_layer=nn.BatchNorm2d):
    # This block applies a downsample of factor = 4 in the scale (x,y)
    block = [nn.Conv2d( input_channels, output_channels, kernel_size=4, stride=2, padding=1)]
    block += [norm_layer(output_channels)]
    block += [nn.LeakyReLU(0.2)]
    return block



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.reflection1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.reflection2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.reflection1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.reflection2(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = x + residual
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_layer=nn.InstanceNorm2d ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(channels)
        self.prelu = nn.LeakyReLU(0.2,inplace=True) #parametric ReLU
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

    
class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=66, repeat_num=6):
        super(Generator, self).__init__()
        initial_layer = [nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1,padding=3, bias=False)]
        initial_layer += [nn.InstanceNorm2d(conv_dim, affine=True)]
        initial_layer += [nn.LeakyReLU(0.2, inplace=True)]

        curr_dim = conv_dim
        for i in range(2):
            initial_layer += [nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)]
            initial_layer += [nn.InstanceNorm2d(curr_dim*2, affine=True)]
            initial_layer += [nn.LeakyReLU(0.2, inplace=True)]
            curr_dim = curr_dim * 2

        self.down_conv = nn.Sequential(*initial_layer)

        bottleneck = []
        for i in range(repeat_num):
            bottleneck += [ResidualBlock(curr_dim)]

        self.bottleneck = nn.Sequential(*bottleneck)

        features = []
        for i in range(2):
            features += [nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)]
            features += [nn.InstanceNorm2d(curr_dim//2, affine=True)]
            features += [nn.LeakyReLU(0.2,inplace=True)]
            curr_dim = curr_dim // 2
        
        self.feature_layer = nn.Sequential(*features)

        colour = [nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)]
        colour += [nn.Tanh()]
        self.colour_layer = nn.Sequential(*colour)

        mask = [nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)]
        mask += [nn.Sigmoid()]
        self.mask_layer = nn.Sequential(*mask)
        init_weights(self)

    def forward( self, x ):
        down = self.down_conv(x)
        bottle = self.bottleneck(down)
        features = self.feature_layer(bottle)
        col = self.colour_layer(features)
        mask = self.mask_layer(features)
        output = mask * ( x[:,0:3,:,:] - col ) + col
        return output


