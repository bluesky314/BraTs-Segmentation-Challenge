
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
   
   
def recall(inpu,target):
    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    recall1=intersection/tar.sum()

    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    recall2=intersection/tar.sum()

    return(recall1,recall2)

def precision(inpu,target):
    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    p1=intersection/ip.sum()
    
    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    p2=intersection/ip.sum()
    
    return(p1,p2)

def get_all_dice(pred_list,target): # for ensemble
    nums=[]
    for e in pred_list:
        s1=1-dice_loss_classes(e, target)[1]
        nums.append(s1)
    return(nums)
def get_all_recall(pred_list,target):
    nums=[]
    for e in pred_list:
        s1=recall(e, target)[1]
        nums.append(s1)
    return(nums)
def get_all_precision(pred_list,target):
    nums=[]
    for e in pred_list:
        s1=precision(e, target)[1]
        nums.append(s1)
    return(nums)

def mean_calc(dict,index):
    l=[]
    for e in dict.keys():
        l.append(dict[e][index])
    return(sum(l)/len(l))
