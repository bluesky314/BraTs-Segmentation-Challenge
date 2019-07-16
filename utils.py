
import numpy as np
 # import nibabel as nib
import io
import torch
from random import shuffle
from scipy.ndimage import rotate
from numpy import percentile
from skimage.transform import resize




def noisy(image): 
    noisy_sigma = np.random.uniform(noise[0],noise[1],1)[0]       
    noisy_smooth=nsmooth 

    row,col,ch= image.shape   
    gauss = np.random.normal(noisy_mean,noisy_sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = nd.gaussian_filter(noisy, sigma=(noisy_smooth, noisy_smooth, noisy_smooth), order=0)

    return noisy

pad=torch.nn.ConstantPad2d(1,0)

def padder(img): 
    
    padded=pad(img.transpose(3,4).transpose(2,3))[:,:,:,:-1,:-1]
    return(padded.transpose(2,3).transpose(3,4))
pad=torch.nn.ConstantPad2d(1,0)

      
def padderw(img): 
    
    
    padded=pad(img)[:,:-1,1:-1] 
    return(padded) 

      
def padderh(img): 
    
    
    padded=pad(img.transpose(0,1))[:,:-1,1:-1]
    return(padded.transpose(0,1)) 
      

def one_hot(a):
    g1=a[:,:,:]==0
    g2=a[:,:,:]==1
    return torch.stack((g1,g2),0)