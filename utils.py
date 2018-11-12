import os
import sys
import time
import torch
import math
import numpy as np
import cv2

def process_image(image,points,angle=0, flip=False, sigma=1,size=128, tight=16):
    if angle > 0:
        if np.random.rand(1) > 0.4:
            tmp_angle = np.random.randn(1) * angle
            image,points = affine_trans(image,points, tmp_angle)
    image, points = crop( image , points, size, tight )
    if flip:
        if np.random.rand(1) > 0.5:
            image,points = flip_ImAndPts(image,points)
        
    image = image/255.0
    image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
    image = image.type_as(torch.FloatTensor())

    source_maps = generate_maps(points, sigma, size)
    source_maps = source_maps.type_as(torch.FloatTensor())   

    return image, source_maps, points


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    point[0] = round( point[0], 2)
    point[1] = round( point[1], 2)

    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image




def generate_maps(points, sigma, size=256):
    maps = None 
    for i in range(0,66):
        tpt = np.array([points[i,0],points[i,1]])
        map = draw_gaussian(np.zeros((size,size)),tpt,sigma=sigma)
        if maps is None:
            maps = torch.from_numpy(map).unsqueeze(0)
        else:
            maps = torch.cat((maps, torch.from_numpy(map).unsqueeze(0)), 0)       
    return maps 

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

def generate_Ginput( img, target_pts , sigma , size=256 ):

    target_maps = generate_maps(target_pts, sigma, size)
    target_maps = target_maps.type_as(torch.FloatTensor())
    A_to_B = torch.cat((img, target_maps),0)  
    return A_to_B



def flip_ImAndPts(image,landmarks):
    flipImg = cv2.flip(image, 1)
    pts_mirror = np.hstack(([range(17,0,-1), range(27,17,-1), range(28,32,1), range(36,31,-1), range(46,42,-1),48,47, range(40,36,-1),42,41,range(55,48,-1),range(60,55,-1),range(63,60,-1),range(66,63,-1)]))
    pts_mirror = pts_mirror - 1
    flipLnd = np.copy(landmarks)
    flipLnd[:,0] = image.shape[1] - landmarks[pts_mirror,0]
    flipLnd[:,1] = landmarks[pts_mirror,1]
    return flipImg,flipLnd

def affine_trans(image,landmarks,angle=None):
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
    dst = cv2.warpAffine(image, M, (nW,nH))
    new_landmarks = np.concatenate((landmarks,np.ones((66,1))),axis=1)
    new_landmarks = new_landmarks.dot(M.transpose())

    return dst, new_landmarks


def gram_matrix(input):
    bsize, ch, r, c = input.size()  
    features = input.view(bsize * ch, r * c) 
    G = torch.mm(features, features.t())  
    return G.div(bsize*ch*r*c)

