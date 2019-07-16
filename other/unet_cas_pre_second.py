

# cant train with bs 1 due to size indexing

alpha,beta,thetha=0.5,0.6,0.6
batch_size = 1
batch_size_val=1
init_lr = 0.003 
log_epoch=0
epoch_magnifiy,epoch_trouble=120,150
log_hot=0

epochs=300; train_size=None
grad_accu_times = 2
if train_size==None: opt_step_size=(270/batch_size)*27 
else:opt_step_size=35*(train_size/batch_size)   

gpu_number = "0"            
noisy_sigma = 0.15    #*********************************************************************         
noisy_smooth=1.5 #*********************************************************************
noisy_mean=0      

batch_norm=False 

leaky_relu=True;relu=False;elu=False 
epoch_val=140 # change wd and add square dice

val_check=24/batch_size_val#*********************************************************************

weight_decay=1e-5 #********************************************************************* 
opt_gamma=0.55           
dropout_toggel,dropblock_toggle=False, True   
dropblock_blocks=3##########################################********************
dropblock_mid=7##########################################********************

dropout_amount=0.3

save_initial='unet_cas_second'
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number
# torch.cuda.set_device(1)
# device = torch.device("cuda:1,2")

# ('input unet', (2, 4, 172, 172, 128))
# ('first down', (2, 4, 86, 86, 64))
# ('Block 1 shape:', (2, 62, 86, 86, 64))
# ('Block 2 shape:', (2, 124, 43, 43, 32))
# ('Block 3 shape:', (2, 248, 21, 21, 16))
# ('Block 4 shape:', (2, 524, 10, 10, 8))
# ('Block Mid shape:', (2, 524, 5, 5, 4))
# ('BlockU 1 shape:', (2, 248, 10, 10, 8))
# ('BlockU 2 shape:', (2, 124, 21, 21, 16))
# ('BlockU 3 shape:', (2, 62, 43, 43, 32))
# ('BlockU 4 shape:', (2, 8, 86, 86, 64))
# ('unet output', (2, 2, 172, 172, 128))
import pickle
from cyclicLR import CyclicCosAnnealingLR
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


brats=pd.read_csv('/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv')
hgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/HGG/'
lgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/LGG/'
csv_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/brats_val/'

def load_obj(name):
    with open('/home/Drive3/rahul/'+ name+ '.pkl', 'rb') as f:
        return pickle.load(f)
dict_train=load_obj('cascadetrain')
dict_val=load_obj('cascadeval')
my_dict = dict_train.copy()
my_dict.update(dict_val)

def classify_label(msk): 
    v=np.zeros((msk.size()[-1],3))
    for i in range(msk.size()[-1]):
        if (msk[:,:,i]==1).any(): v[i,0]=1
        if (msk[:,:,i]==2).any(): v[i,1]=1
        if (msk[:,:,i]==3).any(): v[i,2]=1
    return(torch.tensor(v).transpose(0,1)) # instead of trans can v->(3,size)

def noisy(image): 

    row,col,ch= image.shape   
    gauss = np.random.normal(noisy_mean,noisy_sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = nd.gaussian_filter(noisy, sigma=(noisy_smooth, noisy_smooth, noisy_smooth), order=0)

    return noisy

pad=torch.nn.ConstantPad2d(1,0)


pad=torch.nn.ConstantPad2d(1,0)
def padder(img): 
    
    padded=pad(img.transpose(3,4).transpose(2,3))[:,:,:,:-1,:-1]
    return(padded.transpose(2,3).transpose(3,4))
      
def padderh(img): 
    
    padded=pad(img.transpose(3,4).transpose(2,3))[:,:,:,:-1,1:-1]
    return(padded.transpose(2,3).transpose(3,4))

      
def padderw(img): 
    
    padded=pad(img.transpose(3,4).transpose(2,3))[:,:,:,1:-1,:-1]
    return(padded.transpose(2,3).transpose(3,4))
      
   
def one_hot(a):
    g1=a[:,:,:]==0
    g2=a[:,:,:]==1
    g3=a[:,:,:]==2
    # g4=a[:,:,:]==3
    return torch.stack((g1,g2,g3),0)

seq = iaa.Sequential([  iaa.Fliplr(1),iaa.Flipud(1) ])
seq2 = iaa.Fliplr(1)
seq3 = iaa.Flipud(1)

def read_img(flair,t1,t1ce,t2,seg,eg_name):
            try:flair = np.load(flair)#[15:143,40:212,30:202]
            except: # string error in val 'B' missing from start
                l=[flair,t1,t1ce,t2,seg]
                for i in range(len(l)):
                    s=l[i].split('/')
                    s[-1]='B'+s[-1]
                    l[i]='/'.join(s)
                flair,t1,t1ce,t2,seg=l
                flair=np.load(flair)#[15:143,40:212,30:202]

            t1 = np.load(t1)#[15:143,40:212,30:202]
            t1ce = np.load(t1ce)#[15:143,40:212,30:202]
            t2 = np.load(t2)#[15:143,40:212,30:202]
            seg=np.load(seg)

            
            # seg[seg==4]=3
            # seg=(seg==4)+(seg==2)+(seg==1)+(seg==3) # combine

            seg[seg==2]=0 # remove whole tumor class
            seg[seg==4]=2 
            
            flair=noisy(flair) ; t1=noisy(t1); t1ce=noisy(t1ce); t2=noisy(t2)

            concat_full=np.concatenate([flair,t1,t1ce,t2],axis=0)
            concat_full=torch.tensor(concat_full).transpose(0,1).transpose(1,2).unsqueeze(3)

            h_min,h_max,w_min,w_max,d_min,d_max=my_dict[eg_name][0], my_dict[eg_name][1],my_dict[eg_name][2] ,my_dict[eg_name][3], my_dict[eg_name][4],my_dict[eg_name][5]
            
#             print('dims',h_min,h_max,w_min,w_max,d_min,d_max)
            # print('flair1',np.shape(flair))

            midh=int((h_max+h_min)/2)  # the mid points of wanted region
            midw=int((w_max+w_min)/2) 

            h=h_max-h_min
            w=w_max-w_min
            
            list32=[64,96,32*4,32*5,32*6]
            for i in range(len(list32)):
                if list32[i]-h>0:
                    sel_h=list32[i] # can make this i+1 or condition of greater than n margin
                    break
                    
            for i in range(len(list32)):
                if list32[i]-w>0:
                    sel_w=list32[i]
                    break
            
                    
                    
            sel=max(sel_h,sel_w)
            sel=int(sel/2)
#             print('sizes', sel_h,sel_w,sel)
#             print('indexing', midh-sel,midh+sel, midw-sel,midw+sel)
#             print('mids',midh,midw)
#             print(np.shape(seg),np.shape(concat_full),'concat')
            
#             concat_full=concat_full[ max(0,midh-sel) : min(172,midh+sel), max(0,midw-sel):min(172,midw+sel), :,:]

            # index into the required dimensions
            concat_full2=concat_full[ max(0,midh-sel_h/2) : min(172,midh+sel_h/2), max(0,midw-sel_w/2):min(172,midw+sel_w/2), :,:]
            seg2=seg[max(0,midh-sel_h/2) : min(172,midh+sel_h/2), max(0,midw-sel_w/2):min(172,midw+sel_w/2), :]
#             print(np.shape(concat_full2),'after')
            
            # want dimensions to be div by 4 therefore sub/add to either side until condition is met. In some cases we are at edge so one will not work. sub or add choice is to only include extra region and not cut any
            counter=0
            while np.shape(concat_full2)[0]%4 !=0:
                counter+=1
                concat_full2=concat_full[ max(0,midh-sel_h/2 - counter) : min(172,midh+sel_h/2), max(0,midw-sel_w/2):min(172,midw+ sel_w/2), :,:]
                seg2=seg[ max(0,midh-sel_h/2 - counter) : min(172,midh+sel_h/2), max(0,midw-sel_w/2):min(172,midw+sel_w/2), :]
                if counter==4:break
                    
#             print(np.shape(concat_full2),'s1')
            
            if np.shape(concat_full2)[0]%4 ==0: flag=1
            else:flag=2
                
            counter=0     
            while np.shape(concat_full2)[0]%4 !=0:
                counter+=1
                concat_full2=concat_full[ max(0,midh-sel_h/2 ) : min(172,midh+sel_h/2 + counter), max(0,midw-sel_w/2):min(172,midw+sel_w/2), :,:]
                seg2=seg[ max(0,midh-sel_h/2 ) : min(172,midh+sel_h/2 + counter), max(0,midw-sel_w/2):min(172,midw+sel_w/2), :]
                if counter==4:break
                    
#             print(np.shape(concat_full2),'s2') 

            concat_fullnew=concat_full2
            segnew=seg2
            counter=0    
            
            while np.shape(concat_full2)[1]%4 !=0:
                counter+=1
                concat_full2=concat_fullnew[  : , max(0,midw-sel_w/2 - counter):min(172,midw+sel_w/2), :,:]
                seg2=segnew[  : , max(0,midw-sel_w/2 - counter):min(172,midw+sel_w/2), :]
                if counter==4:break
                    
#             print(np.shape(concat_full2),'s3')        
            counter=0       
            while np.shape(concat_full2)[1]%4 !=0:
                counter+=1
                concat_full2=concat_fullnew[  : , max(0,midw-sel_w/2):min(172,midw+sel_w/2  + counter), :,:]
                seg2=segnew[  : , max(0,midw-sel_w/2):min(172,midw+sel_w/2  + counter), :]
                if counter==4:break
                
#             print(np.shape(seg2),np.shape(concat_full2),'s4')    
                
            concat_full=concat_full2
            seg=seg2
            rand=np.random.choice([0,1,2,3,4])
            # print('seg',np.shape(seg))

            # rand=100
            # if rand==0: 
            #     concat_full=seq.augment_images((np.array(concat_full)))
            #     seg_orig=torch.tensor( seq.augment_images((np.array(seg).astype(int)))      )
            # elif rand==1:
            #     concat_full=seq2.augment_images((np.array(concat_full)))
            #     seg_orig=torch.tensor( seq2.augment_images((np.array(seg).astype(int)))      )
            # elif rand==2:
            #     concat_full=seq3.augment_images((np.array(concat_full)))
            #     seg_orig=torch.tensor( seq3.augment_images((np.array(seg).astype(int)))      )
            # else:
            seg_orig=torch.tensor(seg.astype(int))
            
#             print('seg_orig',np.shape(seg_orig))
            # seg=one_hot(seg_orig) 
#             print('seg_onehot',np.shape(seg))
            

            flair=torch.tensor(concat_full[:,:,:128,0])
            t1=torch.tensor(concat_full[:,:,128:256,0])
            t1ce=torch.tensor(concat_full[:,:,256:384,0])
            t2=torch.tensor(concat_full[:,:,384:512,0])

            img=np.stack([flair,t1,t1ce,t2])
            # print('flair2',np.shape(flair))
        
            seg=one_hot(seg_orig) 
            classify=0

            return(img,seg,seg_orig,classify)



class BraTS_FLAIR_val(torch.utils.data.Dataset):  # Validation dataset getter
 

    def __init__(self, csv_file, root_dir, transform=None): 
        val_dir_hgg=glob.glob('/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/brats_val/HGG/*')
        val_dir_lgg=glob.glob('/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/brats_val/LGG/*')

        self.paths = val_dir_hgg+val_dir_lgg
        self.paths = glob.glob('/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/brats_val/*')
        shuffle(self.paths)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx): 
        
        flair=self.paths[idx]+'/'+str(self.paths[idx])[51:]+'_flairN.npy'  # image files location
        t1=self.paths[idx]+'/'+str(self.paths[idx])[51:]+'_t1N.npy'
        t1ce=self.paths[idx]+'/'+str(self.paths[idx])[51:]+'_t1ceN.npy'
        t2=self.paths[idx]+'/'+str(self.paths[idx])[51:]+'_t2N.npy'        
        seg=self.paths[idx]+'/'+str(self.paths[idx])[51:]+'_seg.npy'
        
        if self.paths[idx][51:54]=='HGG':grade=1
        else: grade=0
        
        print('Val: ',str(self.paths[idx])[51:])
        eg_name=str(self.paths[idx])[51:]
        img,seg,seg_orig,classify=read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg,eg_name=eg_name)

        sup_block3=resize(np.array(seg_orig),[np.shape(seg_orig)[0]/2, np.shape(seg_orig)[1]/2, 128/2],order=0,mode='constant', preserve_range=True)
        sup_block3=torch.tensor(sup_block3) 
        one_hot_b3=one_hot(sup_block3)#.transpose(0,1) 

        sup_block4=resize(np.array(seg_orig),[172/2, 172/2, 128/2],order=0,mode='constant', preserve_range=True)
        sup_block4=torch.tensor(sup_block4)
        one_hot_b4=one_hot(sup_block4)#.transpose(0,1) 

        sample = {'img': torch.from_numpy(img), 'mask': seg.type(torch.ByteTensor),'seg_orig':seg_orig.type(torch.ByteTensor),'eg_name':eg_name,'classify':classify,
        'sup_block3':sup_block3,'one_hot_b3':one_hot_b3,'sup_block4':sup_block4,'one_hot_b4':one_hot_b4,'grade':torch.tensor(grade)}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class BraTS_FLAIR(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, transform=None,train_size=train_size):  
        self.root_dir = root_dir
        self.transform = transform

        self.hggs=glob.glob(hgg_dir+'*')
        self.lggs=glob.glob(lgg_dir+'*')
        self.brats=self.hggs+self.lggs

        shuffle(self.brats)

        if train_size: self.brats=self.brats[:train_size]  #  chosen subset
        
          

    def __len__(self):                                    
        return len(self.brats)

    def __getitem__(self, idx): 
        folder=self.brats[idx]

        flair=self.brats[idx]+'/'+self.brats[idx][45:]+'_flairN.npy'  # image files location
        t1=self.brats[idx]+'/'+self.brats[idx][45:]+'_t1N.npy'
        t1ce=self.brats[idx]+'/'+self.brats[idx][45:]+'_t1ceN.npy'
        t2=self.brats[idx]+'/'+self.brats[idx][45:]+'_t2N.npy'
        seg=self.brats[idx]+'/'+self.brats[idx][45:]+'_seg.npy'
        
        if self.brats[idx][41:44]=='HGG': grade=1
        else: grade=0
        
        
        print('Train: ',self.brats[idx][45:])
        eg_name=self.brats[idx][45:]
        

        img,seg,seg_orig,classify= read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg,eg_name=eg_name)

        sup_block3=resize(np.array(seg_orig),[np.shape(seg_orig)[0]/2, np.shape(seg_orig)[1]/2, 128/2],order=0,mode='constant', preserve_range=True)
        sup_block3=torch.tensor(sup_block3) 
        one_hot_b3=one_hot(sup_block3)#.transpose(0,1)#.squeeze(0) 

        sup_block4=resize(np.array(seg_orig),[172/2, 172/2, 128/2],order=0,mode='constant', preserve_range=True)
        sup_block4=torch.tensor(sup_block4)
        one_hot_b4=one_hot(sup_block4)#.transpose(0,1)#.squeeze(0) 
        # print(sup_block3.size(),one_hot_b3.size())
        sample = {'img': torch.from_numpy(img), 'mask': seg.type(torch.ByteTensor),'seg_orig':seg_orig.type(torch.ByteTensor),'eg_name':eg_name,'classify':classify,
        'sup_block3':sup_block3,'one_hot_b3':one_hot_b3,'sup_block4':sup_block4,'one_hot_b4':one_hot_b4,'grade':torch.tensor(grade)}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

 
 
 
brats=pd.read_csv('/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv')
hgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/HGG/'
lgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/LGG/'
csv_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Pre/brats_val/'


class my_norm(torch.nn.Module):
    def __init__(self, channels):
        super(my_norm,self).__init__()
        self.norm=torch.nn.BatchNorm3d(channels) if batch_norm==True else torch.nn.InstanceNorm3d(channels, affine=True)

    def forward(self,x): return(self.norm(x))


class my_activation(torch.nn.Module):
    def __init__(self):
        super(my_activation,self).__init__()
        if leaky_relu:
            self.activation=torch.nn.LeakyReLU(inplace=True) 
        if relu:
            self.activation=torch.nn.ReLU(inplace=True) 
        if elu:
            self.activation=torch.nn.ELU(inplace=True) 
    def forward(self,x):return(self.activation(x))



class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=1)
        self.bn1 = my_norm(output_channel) 
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = my_norm(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = my_norm(output_channel)
        self.max_pool = torch.nn.MaxPool3d(2, 2)
        self.relu = my_activation()
        self.down_size = down_size
        if dropout_toggel==True: self.dropout=torch.nn.Dropout3d(dropout_amount)
        if dropblock_toggle==True: self.dropout=LinearScheduler(DropBlock3D(block_size=dropblock_blocks, drop_prob=0.),
                start_value=0.,
                stop_value=0.25,
                nr_steps=5)


    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.dropout(self.bn1(self.relu(self.conv1(x))))
        residual=x.clone()
        x = self.dropout(self.bn2(self.relu(self.conv2(x))))
        x = self.dropout(self.bn3(self.relu(self.conv3(x))))
        x=x+residual
        return x

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
#         self.up_sampling = torch.nn.functional.interpolate(scale_factor=2, mode='trilinear')
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = my_norm(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = my_norm(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = my_norm(output_channel)
        self.relu = my_activation()
        if dropout_toggel==True: self.dropout=torch.nn.Dropout3d(dropout_amount)
        if dropblock_toggle==True: self.dropout=LinearScheduler(DropBlock3D(block_size=dropblock_blocks, drop_prob=0.),
                start_value=0.,
                stop_value=0.25,
                nr_steps=5)

        self.trans=torch.nn.ConvTranspose3d(input_channel,input_channel,2,2,padding=0)

    def forward(self, prev_feature_map, x):
        # print('before trans',x.size())
        x=self.trans(x)
        # print('upblock',x.size(),prev_feature_map.size())
        try:
            x = torch.cat((x, prev_feature_map), dim=1)
        except:
            if x.size()[2] != prev_feature_map.size()[2]:
                x=padderh(x)
#                 print('upblock2', x.size(),prev_feature_map.size())
            if x.size()[3] != prev_feature_map.size()[3]:
                x=padderw(x)
#                 print('upblock3', x.size(),prev_feature_map.size())
#             print('upblock4', x.size(),prev_feature_map.size())
            # print('upblock',x.size(),prev_feature_map.size())
            x = torch.cat((x, prev_feature_map), dim=1)
        
        x = self.dropout(self.bn1(self.relu(self.conv1(x))))
        residual=x.clone()
        x = self.dropout(self.bn1(self.relu(self.conv2(x))))
        x = self.dropout(self.bn1(self.relu(self.conv3(x))))
        x=x+residual
        return x


class UNet(torch.nn.Module):        # input -> shrunk in 1/2 then transposed at end. Loss can be with orig seg label / shurken if remove transpose
    def __init__(self):
        super(UNet, self).__init__()

        self.preconv=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb = my_norm(4)
        self.preconv2=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb2 = my_norm(4)
        self.preconv3=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb3 = my_norm(4)

        self.firstconv=torch.nn.Conv3d(4,4,3,stride=2,padding=1)
        # self.bn0 = torch.nn.BatchNorm3d(4)
        big=524

        self.down_block1 = UNet_down_block(4, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 248, True)
        self.down_block4 = UNet_down_block(248, big, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)

        self.mid_conv1 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(big)
        self.mid_conv2 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(big)
        self.mid_conv3 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(big)

        self.up_block1 = UNet_up_block(big, big, 248)
        self.up_block2 = UNet_up_block(248, 248, 128)
        self.up_block3 = UNet_up_block(128, 128, 64)
        self.up_block4 = UNet_up_block(64, 64, 8)


        self.last_conv1 = torch.nn.Conv3d(12, 8, 3, padding=1) # 12->8 no input concat
        self.last_bn = my_norm(8)
        self.last_conv2 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn2= my_norm(4)
        self.relu = my_activation()
        self.last_conv3 = torch.nn.Conv3d(4, 3, 3, padding=1)#made 4-> 2


        self.trans=torch.nn.ConvTranspose3d(8,8,2,2,padding=0)
        self.convt = torch.nn.Conv3d(8, 8, 3, padding=1)
        self.bnt= my_norm(8)

        self.dsb3 = torch.nn.Conv3d(64, 3, 3, padding=1)#made 4-> 2
        self.dsb4 = torch.nn.Conv3d(8, 3, 3, padding=1)#made 4-> 2
        
        # self.class1 = torch.nn.Conv3d(12, 6, 3, padding=1)
        # self.classbn=my_norm(6)
        # self.class2 = torch.nn.Conv3d(6, 3, 3, padding=1)
        # self.avg=torch.nn.AdaptiveMaxPool3d((1,1,128))
        self.avg1=torch.nn.AdaptiveMaxPool3d((1,1,1))
        
        self.gradeconv=torch.nn.Linear(904,100)
        self.gradeconv2=torch.nn.Linear(100,50)
        self.gradeconv3=torch.nn.Linear(50,1)
         
    def forward(self, x):
        print('input unet',x.size())
        inputs = x.clone()

        x = self.preconvb(self.relu(self.preconv(x)))
        x = self.preconvb2(self.relu(self.preconv2(x)))
        x = self.preconvb3(self.relu(self.preconv3(x)))

        x=self.relu(self.firstconv(x))
        # print('first down',x.size())

        self.x1 = self.down_block1(x)
        # print("Block 1 shape:",self.x1.size())
        self.x2 = self.down_block2(self.x1)
    
        # print("Block 2 shape:",self.x2.size())
        self.x3 = self.down_block3(self.x2)
        # print("Block 3 shape:",self.x3.size())
    
        self.x4 = self.down_block4(self.x3)
        # print("Block 4 shape:",self.x4.size())
    
        self.xmid=self.max_pool(self.x4)
        self.xmid = self.bn1(self.relu(self.mid_conv1(self.xmid)))
        residual=self.xmid.clone()
        self.xmid = self.bn2(self.relu(self.mid_conv2(self.xmid)))
        self.xmid = self.bn3(self.relu(self.mid_conv3(self.xmid)))
        self.xmid = self.xmid + residual
        # print("Block Mid shape:",self.xmid.size())
        
        b1 = self.up_block1(self.x4, self.xmid)
        # print("BlockU 1 shape:",x.size())
        b2 = self.up_block2(self.x3, b1)
        # print("BlockU 2 shape:",x.size())
        
        b3 = self.up_block3(self.x2, b2)
        # print("BlockU 3 shape:",b3.size())
 
        b4 = self.up_block4(self.x1, b3) #bs,8,86,86,64
        # print("BlockU 4 shape:",b4.size())
        
        x=  self.trans(b4)   #bs,8,172,172,128  #kernal size is 2
        x = self.bnt(self.relu(self.convt(x)))
        # print('ins',x.size(),inputs.size())
        # print('ins',padderw(x).size(),inputs.size())
        # print('ins',padderh(x).size(),inputs.size())
        try:
            x = torch.cat((x, inputs), dim=1) # #bs,12,172,172,128 concat with input  * show x not inps
        except:
            try:x = torch.cat((padderw(x), inputs), dim=1)
            except:x = torch.cat((padderh(x), inputs), dim=1)

        
        # classify=self.classbn(self.relu(self.class1(x)))
        # classify=self.class2(classify) #bs,3,172,172,128 
        # pool=self.avg(classify).squeeze(2).squeeze(2)# bs,3,1,1,128->bs,3,128 

        
        
        x = self.last_bn(self.relu(self.last_conv1(x)))
        x = self.last_bn2(self.relu(self.last_conv2(x)))
        x = self.last_conv3(x)
        
        b3=self.dsb3(b3) # supervision
        b4=self.dsb4(b4)
        # print('unet output',x.size())

        # grader=torch.cat( (self.avg1(self.xmid),self.avg1(b1),self.avg1(b2),self.avg1(b3),self.avg1(b4)),1 ).squeeze(-1).squeeze(-1).squeeze(-1)

        # grader=self.relu(self.gradeconv(grader))
        # grader=self.relu(self.gradeconv2(grader))
        # grader=self.relu(self.gradeconv3(grader))

        return(0,b3,b4,x)




def dice_loss(inpu, target):  # [bs,4,96,96,64]
    a=0;b=0;c=0;d=0 
    intersection=0; union=0
    

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection+=(ip * tar).sum() 
    union+= ip.sum() + tar.sum()
     
    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection+=(ip * tar).sum() 
    union+= ip.sum() + tar.sum()
   
 
    raw_scores=1-2*(intersection/union)
    
    return raw_scores



def dice_loss_classes(inpu, target):  # [bs,4,96,96,64]

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
    
    ip=inpu[:,2,:,:,:].contiguous().view(-1)
    tar=target[:,2,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score3=1-2*(intersection/union)
     
    
    return score1,score2,score3


def dice_loss_val(inpu, target):
    a=0;b=0;c=0;d=0 

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    a+=calc_val(ip,tar)
     
    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    b+=calc_val(ip,tar)
     
    raw_scores=( a +  b )/2
    
    return raw_scores


def square_dice(a): return(10*a*a)
L=np.array([]);Lc=np.array([]);Ld=np.array([]);Lv=np.array([]);Lvd=np.array([]);Lvc=np.array([]);L_r=np.array([]);Ldc=np.array([]);Lvdc=np.array([])


model = UNet().cuda()
# model.load_state_dict(torch.load('/home/Drive3/rahul/unet_cas_pre-0.28|18')['state_dict'])


hotlist={}
biglist={}
def train(): 
    global Lc,Lvc,Lvdc, Lc,Ldc,L,Lv
    forward_times=0
    counter_t=0
    counter_v=0
    vis_count=0

    dataset = BraTS_FLAIR(csv_dir,hgg_dir,transform=None,train_size=train_size)
    dataset_val=BraTS_FLAIR_val(csv_dir,hgg_dir)    #val
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    val_loader=DataLoader(dataset_val,batch_size_val,shuffle=True, num_workers=2) 
    loaders={'train':data_loader,'val': val_loader}

 
    top_models = [(0, 10000)] * 5 # (epoch,loss)
    worst_val = 10000 # init val save loop

    loss_fn = torch.nn.CrossEntropyLoss()  
    soft_max=torch.nn.Softmax(dim=1)
    bce_classify = torch.nn.BCELoss() 
    sigmoid=torch.nn.Sigmoid()

    
    opt = torch.optim.Adam(model.parameters(), lr=init_lr,weight_decay=weight_decay)
    scheduler = StepLR(opt, step_size=opt_step_size, gamma=opt_gamma)  #**
    # multisteplr 0.5,

    opt.zero_grad()

    for epoch in range(epochs):
        for e in loaders:
            if e=='train':  model.train() ; grad=True #
            else: model.eval() ; grad=False

            loss_v=0; count_v=0
            with torch.set_grad_enabled(grad):    
                for idx, batch_data in enumerate(loaders[e]):
                    if e=='train': counter_t+=1
                    else: counter_v+=1; count_v+=1
                    torch.cuda.empty_cache()
                    batch_input = Variable(batch_data['img'].float()).cuda()#.to(device)
                    batch_gt_mask = Variable(batch_data['mask'].float()).cuda()#.to(device)
                    batch_seg_orig=Variable(batch_data['seg_orig']).cuda()#.to(device)
                    eg_name=batch_data['eg_name']
                    # classify_tars=batch_data['classify'].cuda()
                    sup_block3=Variable(batch_data['sup_block3']).cuda()
                    one_hot_b3=Variable(batch_data['one_hot_b3']).cuda()
                    sup_block4=Variable(batch_data['sup_block4']).cuda()
                    one_hot_b4=Variable(batch_data['one_hot_b4']).cuda()
                    grades=Variable(batch_data['grade']).cuda()
                    
                    temp_size=batch_gt_mask.size()

                    valid=True

                    try:
                        grader,b3,b4,pred_mask = model(batch_input)
                    except:
                        print('*********************************skip************************')
                        valid=False
                        pred_mask=torch.rand(temp_size).cuda()


                    if e=='train':forward_times += 1

                    ce = loss_fn(pred_mask, batch_seg_orig.long())
                    soft_mask=soft_max(pred_mask)
                    dice=dice_loss(soft_mask, batch_gt_mask)


                    a,b,c=dice_loss_classes(soft_mask, batch_gt_mask)
                    # ad=a
                    # bd=b
                    # if epoch>epoch_val: ad=square_dice(a); bd=square_dice(b)
                    
                    lossnet = ce+(a+b+c)/3 # +dice
                    
                    if valid: loss_v+=(b+c)/2

                    # if b3.size()[2] != sup_block3.size()[1]:  # as sizes are var, we have to pad. supblock3 is of CE form of target hence it needs one less index
                    #     b3=padderh(b3)
                    # if b3.size()[3] != sup_block3.size()[2]:
                    #     b3=padderw(b3)
                    # # print('lb3',b3.size(),sup_block3.size())

                    # ceb3 = loss_fn(b3, sup_block3.long())
                    # soft_mask_b3=soft_max(b3)
                    # a1,b1=dice_loss_classes(soft_mask_b3, one_hot_b3.float())
                    # if epoch>epoch_val: a1=square_dice(a1); b1=square_dice(b1)
                    # lossb3=ceb3+(a1+b1+c1)/3
                    


                    # ceb4 = loss_fn(b4, sup_block4.long())
                    # soft_mask_b4=soft_max(b4)
                    # a2,b2=dice_loss_classes(soft_mask_b4, one_hot_b4.float())
                    # if epoch>epoch_val: a2=square_dice(a2); b2=square_dice(b2)
                    # lossb4=ceb4+(a2+b2)/2
                    

                    # loss_classify=bce_classify(sigmoid(classify),classify_tars.float())
                    # loss_grader=bce_classify(sigmoid(grader),grades.float())
                    # print('--------------------------------------------------------------------',loss_grader.item())
                    loss=lossnet  #+  alpha*lossb3 #+  beta*lossb4 
                    # if epoch>100: loss += thetha*loss_grader

                    if valid: print('Dice Losses: ',a.item(),b.item(),c.item())
                    if e=='train': 
                        # if epoch>epoch_magnifiy:
                        #     if b>0.18 or a>0.18 :
                        #         loss=3*loss
                        #         print('Magnified')

                        if epoch>epoch_trouble:
                            if loss>1.5*loss_moving_avg:
                                loss=3*loss
                                print('Trouble')
                        
                        

                        if valid: loss.backward()
                        
                        if forward_times == grad_accu_times:
                            if valid:
                                Lc=np.concatenate((np.array([ce.item()]), Lc)); cross_moving_avg=np.mean(Lc)
                                Ldc=np.concatenate((np.array([b.item()]), Ldc)); dice1=np.mean(Ldc)
                            
                                L=np.concatenate((np.array([loss.item()]), L)); loss_moving_avg=np.mean(L)
                                print('Epoch: ', epoch+1, ' Batch: ',idx+1,' lr: ',scheduler.get_lr()[-1], ' Dice1: ', dice1, ' CE: ',cross_moving_avg,' Loss :' ,loss_moving_avg)

                                opt.step()
                                opt.zero_grad()
                                forward_times = 0
                                print('\nUpdate weights ... \n')
                        if valid:
                            writer.add_scalar('Total Train Loss', loss.item(), counter_t)
                            writer.add_scalar('Target Train Loss', lossnet.item(), counter_t)
                            writer.add_scalar('Train CE', ce.item(), counter_t)
                            writer.add_scalar('D3', c.item(), counter_t)
                            # writer.add_scalar('loss_grader', loss_grader.item() , counter_t)
                            writer.add_scalar('D1', a.item(), counter_t)
                            writer.add_scalar('D2', b.item(), counter_t)
                            # writer.add_scalar('D3', c.item() , counter_t)

                            # writer.add_scalar('D4', d.item() , counter_t)
                            # writer.add_scalar('lossb3', lossb3.item() , counter_t)
                            # writer.add_scalar('lossb4', lossb4.item() , counter_t)
                            writer.add_scalar('Lr', scheduler.get_lr()[-1] , counter_t)
                        # writer.add_scalar('Lr', lz , counter_t)


                        torch.cuda.empty_cache()
                        if valid:
                            if epoch>log_hot:
                                if b>0.25 or c>0.25 or eg_name[0] in hotlist.keys(): #hotlist,start after N epochs,keep tracking once added
                                    print('hotlist culprit')
                                    if eg_name[0] not in hotlist.keys():
                                        hotlist[eg_name[0]]=[ [ce.item()] , [b.item()]]
                                    else:
                                        hotlist[eg_name[0]][0].append(ce.item())
                                        hotlist[eg_name[0]][1].append(b.item())

                                        # hotlist[eg_name[0]][2].append(c.item())
                                        # hotlist[eg_name[0]][3].append(d.item())
                            if epoch>log_epoch:
                                if epoch%10==0:
                                    print('biglist updated')

                                    if eg_name[0] not in biglist.keys():
                                        biglist[eg_name[0]]=[ [ce.item()] , [b.item()] ]
                                    else:
                                        biglist[eg_name[0]][0].append(ce.item())
                                        biglist[eg_name[0]][1].append(b.item())
                                        # biglist[eg_name[0]][2].append(c.item())
                                        # biglist[eg_name[0]][3].append(d.item())

                            
                            # vis(soft_mask,batch_seg_orig,vis_count,mode='train')
                            # vis_count+=25 # += no of images in vis loop
                            scheduler.step()


                    else:
                        if valid:
                            Lv=np.concatenate((np.array([loss.item()]), Lv)) ; lv_avg=np.mean(Lv)
                            Lvc=np.concatenate((np.array([ce.item()]), Lvc)) ;lvc_avg=np.mean(Lvc)
                            Lvdc=np.concatenate((np.array([b.item()]), Lvdc)); 
                        
                        # if epoch==epoch_val:
                        #     opt = torch.optim.Adam(model.parameters(), lr=scheduler.get_lr()[-1],weight_decay=weight_decay*3)#continue from last lr with *3 decay
                        #     scheduler = StepLR(opt, step_size=(270/batch_size)*15 , gamma=opt_gamma)  # AND quicken lr decrease 
    # multisteplr 0.5,
                            ###tester#####
                            # print('----lr test--------------')
                            # c= 5  # currently 1 val for each epoch. change, remove break at end val_size/batch_size_val * 10

                            # print(np.shape(Lvdc),np.mean(Lvdc[:-c]),np.mean(Lvdc[:]))
                            # if np.mean(Lvdc[:-c])-np.mean(Lvdc[:]) <0.01:
                            #     for param_group in opt.param_groups:
                            #         param_group['lr'] = param_group['lr']/3

                            #         print('---------------------------------------changed lr/2-----------------------------------------------')
                            ###tester#####


                        if valid:
                            writer.add_scalar('Val Loss', loss.item(), counter_v)
                            writer.add_scalar('Val CE', ce.item(), counter_v)
                            # writer.add_scalar('Val Dice', dice.item(), counter_v)
                            writer.add_scalar('D1v', a.item(), counter_v)
                            writer.add_scalar('D2v', b.item(), counter_v)
                            writer.add_scalar('D3v', c.item(), counter_v)
                            # writer.add_scalar('loss_graderV', loss_grader.item() , counter_v)
                            # vis(soft_mask,batch_seg_orig,vis_count,mode='val')
                            print(save_initial)

                            print('Validation total Loss::::::::::', round(loss.item(),3) )
                            del batch_input, batch_gt_mask, batch_seg_orig, pred_mask

                            print('current n worst val: ',round(loss.item(),2),worst_val)
                            # print('------------------------------',count_v,val_check)
                            if count_v%val_check==0:
                                if epoch>epoch_val+15:
                                    for param_group in opt.param_groups: # cant get lr from cheduler
                                        param_group['weight_decay']=weight_decay*3
                                #         scheduler = StepLR(opt, step_size=opt_step_size/2, gamma=0.6)



                                loss=loss_v/val_check # loss_v is total loss of val, val_check is no of batches, hence loss is avg loss of target only(no ds)


                                print('------------------val mean-----------', loss)
                                if epoch>70 and epoch%20==0: # save every 15- for logs- confilicting with down

                                    checkpoint = {'epoch': epoch + 1, 'moving loss':L,'val':Lv,'hotlist':hotlist,'biglist':biglist,
                                    'Ldc':Ldc, 'cross el':Lc,'Ldvc':Lvdc, 'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }

                                    torch.save(checkpoint, '/home/Drive3/rahul/'+save_initial +'-' + str(round(loss,2))+'|'+str(epoch+1) + 'r')
                                    print('Saved at 25 : ', save_initial +'-' +str(round(loss,2))+'|'+str(epoch+1))


                                if loss<worst_val : 
                                    # print('saving --------------------------------------',epoch)
                                    top_models=sorted(top_models, key=lambda x: x[1]) # sort maybe not needed 
                                    
                                    checkpoint = {'epoch': epoch + 1, 'moving loss':L,'val':Lv,'hotlist':hotlist,'biglist':biglist,
                                    'Ldc':Ldc, 'cross el':Lc,'Ldvc':Lvdc, 'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }
                                    torch.save(checkpoint, '/home/Drive3/rahul/'+save_initial +'-' +str(round(loss,2))+'|'+str(epoch+1))

                                    to_be_deleted='/home/Drive3/rahul/'+ save_initial +'-' +str(round(top_models[-1][1],2))+'|'+str(top_models[-1][0]) # ...loss|epoch
                                    # print(to_be_deleted)

                                    top_models.append((epoch+1,loss.item()))
                                    
                                    top_models=sorted(top_models, key=lambda x: x[1]) #sort after addition of new val
                                    top_models.pop(-1)

                                    print('top_models',top_models)

                                    worst_val=top_models[-1][1]
                                    if str(to_be_deleted)!='/home/Drive3/rahul/'+save_initial +'-' +'10000.0|0':     # first 5 epoch will be saved and no deletion this point
                                        os.remove(to_be_deleted)
                                    # print('sucess deleted------------------')
                                    
                        
print('here')
train()
        

    
