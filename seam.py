import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import choice
from scipy import ndimage
import scipy
from PIL import Image
import random


def gradient(img, x, y):
    gx = img[x+1,y] - img[x,y]
    gy = img[x,y+1] - img[x,y]
    return np.sqrt(gx**2+gy**2)

def smooth_difference(img1, img2, x, y, alpha):
    d = (img1[x,y] - img2[x,y])**2 + alpha*((gradient(img1, x, y) - gradient(img2, x, y))**2)
    return d

def neighbourhood(img, x, y):
    neigh = []
    if(x-1>=0):
        if(y-1>=0):
            neigh.append([x-1,y-1])
        neigh.append([x-1,y])
    if(x+1<img.shape[0]):
        if(y+1<img.shape[1]):
            neigh.append([x+1,y+1])
        neigh.append([x+1,y])
    return neigh

def energy(img1, img2, lambd, alpha, x, y):
    Ed1 = -gradient(img1, x, y)
    Ed2 = -gradient(img2, x, y)
    nei1 = neighbourhood(img2, x, y)
    nei2 = neighbourhood(img1, x, y)
    Es1 = 0
    Es2 = 0
    for [nx,ny] in nei1:
        Es1+=smooth_difference(img1, img2, x, y, alpha) + smooth_difference(img1, img2, nx, ny, alpha)
    for [nx,ny] in nei2:
        Es2+=smooth_difference(img1, img2, x, y, alpha) + smooth_difference(img1, img2, nx, ny, alpha)    
    E1 = Ed1 + lambd*Es1
    E2 = Ed2 + lambd*Es2
    if E1<E2:
        return img1[x,y]
    return img2[x,y]