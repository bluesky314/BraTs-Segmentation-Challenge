import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import io
import torch
import torch.utils#.data.Dataset
import glob
import imgaug as ia
from imgaug import augmenters as iaa


def dice_loss_classes(inpu, target):  
    ''' Dice Loss for two classes '''

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score1=1-2*(intersection/union)
    

    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score2=1-2*(intersection/union)
     
    
    return score1,score2


def tversky_loss(inpu, target,alpha,beta):  
    ''' Tversky Loss for two classes ''' 

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    
    fps = torch.sum(inpu * (1 - target))
    fns = torch.sum((1 - inpu) * target)
    num = intersection
    
    denom = intersection + (alpha * fps) + (beta * fns)
    
    score1=1-2*(intersection/denom)
    

    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    
    fps = torch.sum(inpu * (1 - target))
    fns = torch.sum((1 - inpu) * target)
    num = intersection
    
    denom = intersection + (alpha * fps) + (beta * fns)
    
    score2=1-2*(intersection/denom)
     
    
    return score1,score2


def power_dice(x,alpha,n): 
    ''' Power Loss'''
    
    return(alpha*(x**n))

