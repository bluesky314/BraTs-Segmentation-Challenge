alpha,beta=0.6,0.7 #weights for supervsion in b3,b4
batch_size = 1
batch_size_val=1
init_lr = 0.025
log_epoch=100
epoch_magnifiy,epoch_trouble=130,170


epochs=800; train_size=None   # Set to None to use all     # DataLoader: train size
grad_accu_times = 2
if train_size==None: opt_step_size=(270/batch_size)*25 #15*125  # scheduler=for StepLR
else:opt_step_size=40*(train_size/batch_size)   

gpu_number = "0"            # which gpu to run on( as str 0-2)
noisy_sigma = 0.003             # gaussian noise std
noisy_smooth=1.5   # gaussian_filter smooth
noisy_mean=0      # gaussian_filter mean
standardizer_type=['mode'] # options: 'he' and ('mode' or mean' or 'median')  override order

batch_norm=False # If False instance norm is taken

leaky_relu=True;relu=False;elu=False # activations in overridding order

# #((43, 43, 32), (2, 43, 43, 32))
# ('Train: ', 'Brats18_TCIA01_412_1')
# ('Train: ', 'Brats18_TCIA01_411_1')
# ((1, 43, 43, 32), (1, 2, 43, 43, 32))


weight_decay=0.001
opt_gamma=0.55           # scheduler gamma
dropout_toggel,dropblock_toggle=False, True   # only one true, dropblock overrides dropout
dropblock_blocks=5
dropblock_mid=9

dropout_amount=0.3

save_initial='unet_casw'
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number
# torch.cuda.set_device(1)
# device = torch.device("cuda:1,2")




from DropBlock3d import DropBlock2D, DropBlock3D,LinearScheduler
import gzip
from sh import gunzip
import numpy as np
import pandas as pd
import gzip
import nibabel as nib
import matplotlib.pyplot as plt
import io
import torch
import torch.utils#.data.Dataset
import glob
#from sklearn.preprocessing import OneHotEncoder 
import imgaug as ia
from imgaug import augmenters as iaa
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as nd
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import warnings
warnings.filterwarnings('ignore')
from random import shuffle
from scipy.ndimage import rotate
from numpy import percentile
from skimage.transform import resize



import time
from torch.optim.lr_scheduler import StepLR
import torch as torch
from skimage import exposure
from tensorboardX import SummaryWriter
writer = SummaryWriter()


hgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/HGG/'
lgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/LGG/'
csv_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/'
names=[]

l1=[];l2=[];l3=[];l4=[]
l1v=[];l2v=[];l3v=[];l4v=[]

l1mode=[];l2mode=[];l3mode=[];l4mode=[]
l1modev=[];l2modev=[];l3modev=[];l4modev=[]

hggs=glob.glob(hgg_dir+'*')
lggs=glob.glob(lgg_dir+'*')
brats=hggs+lggs
for idx in range(len(brats)):
    folder= brats[idx]

    flair= brats[idx]+'/'+ brats[idx][55:]+'_flairB.npy'  # image files location
    t1= brats[idx]+'/'+ brats[idx][55:]+'_t1B.npy'
    t1ce= brats[idx]+'/'+ brats[idx][55:]+'_t1ceB.npy'
    t2= brats[idx]+'/'+ brats[idx][55:]+'_t2B.npy'
    seg= brats[idx]+'/'+ brats[idx][55:]+'_seg.nii'

    try:flair = np.load(flair)[40:212,30:202,15:143]
    except: # string error in val 'B' missing from start
        l=[flair,t1,t1ce,t2,seg]
        for i in range(len(l)):
            s=l[i].split('/')
            s[-1]='B'+s[-1]
            l[i]='/'.join(s)
        flair,t1,t1ce,t2,seg=l
        flair=np.load(flair)[40:212,30:202,15:143]

    t1 = np.load(t1)[40:212,30:202,15:143]
    t1ce = np.load(t1ce)[40:212,30:202,15:143]
    t2 = np.load(t2)[40:212,30:202,15:143]
    
    mod=0         
    imgs=[flair,t1,t1ce,t2]
    for img in imgs:
        mod+=1
        img=torch.tensor(img)

        z=img.flatten()[img.flatten()>0]

        mean=z.mean()
        mode=z.float().mode()[0]

        mean_var=(z-mean)**2
        mean_var=mean_var.sum()

        mode_var=(z-mode)**2
        mode_var=mode_var.sum()

        print(mod,mean,mode)

        if mod==1:
            l1.append(mean.item())
            l1v.append(mean_var.item())
            l1mode.append(mode.item())
            l1modev.append(mode_var.item())
        if mod==2:
            l2.append(mean.item())
            l2v.append(mean_var.item())
            l2mode.append(mode.item())
            l2modev.append(mode_var.item())
        if mod==3:
            l3.append(mean.item())
            l3v.append(mean_var.item())
            l3mode.append(mode.item())
            l3modev.append(mode_var.item())
        if mod==4:
            l4.append(mean.item())
            l4v.append(mean_var.item())
            l4mode.append(mode.item())
            l4modev.append(mode_var.item())

    print('------------------',idx,'---------------------------------')
    # if idx%10==0:
    #     print(l1)
    #     print(l2)
    #     print(l3)
    #     print(l4)

np.save('/home/Drive3/rahul/means',np.array([l1,l2,l3,l4]))

np.save('/home/Drive3/rahul/means_var',np.array([l1v,l2v,l3v,l4v]))

np.save('/home/Drive3/rahul/modes',np.array([l1mode,l2mode,l3mode,l4mode]))

np.save('/home/Drive3/rahul/modes_var',np.array([l1modev,l2modev,l3modev,l4modev]))