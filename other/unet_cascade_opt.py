

# data is shuffled before indexed by train_size to include some lgg

alpha,beta=0.5,0.6
batch_size = 1
batch_size_val=1
init_lr = 0.003
log_epoch=180
epoch_magnifiy,epoch_trouble=140,180
log_hot=180

epochs=600; train_size=6   
grad_accu_times = 2
if train_size==None: opt_step_size=(270/batch_size)*25 
else:opt_step_size=35*(train_size/batch_size)   

gpu_number = "1"            
noisy_sigma = 0.08            
noisy_smooth=1.5   
noisy_mean=0      

batch_norm=False 

leaky_relu=True;relu=False;elu=False 


weight_decay=1e-4
opt_gamma=0.55           
dropout_toggel,dropblock_toggle=False, True   
dropblock_blocks=5
dropblock_mid=9

dropout_amount=0.3

save_initial='unet_cas_med'
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


def standardizer(img,mod):

    if mod==0:mean=233.581878; std=204.35788
    elif mod==1:mean=599.1488; std=547.67474
    elif mod==2:mean=506.3251647; std=533.85250
    else: mean=388.70568; std=378.741204

    img=img-mean
    img=img/std
   
    quartiles = percentile(img, [99.5])  # exact region
    quartiles2 = percentile(img, [0.05]) 

    img=np.clip(img,quartiles2[0],quartiles[0])

    return(img)

pad=torch.nn.ConstantPad2d(1,0)
def padder(img): 
    
    padded=pad(img.transpose(3,4).transpose(2,3))[:,:,:,:-1,:-1]
    return(padded.transpose(2,3).transpose(3,4))
      

def one_hot(a):
    g1=a[:,:,:]==0
    g2=a[:,:,:]==1
    # g3=a[:,:,:]==2
    # g4=a[:,:,:]==3
    return torch.stack((g1,g2),0)

seq = iaa.Sequential([  iaa.Fliplr(1),iaa.Flipud(1) ])
seq2 = iaa.Fliplr(1)
seq3 = iaa.Flipud(1)

def read_img(flair,t1,t1ce,t2,seg):
            try:flair = np.load(flair)[15:143,40:212,30:202]
            except: # string error in val 'B' missing from start
                l=[flair,t1,t1ce,t2,seg]
                for i in range(len(l)):
                    s=l[i].split('/')
                    s[-1]='B'+s[-1]
                    l[i]='/'.join(s)
                flair,t1,t1ce,t2,seg=l
                flair=np.load(flair)[15:143,40:212,30:202]

            t1 = np.load(t1)[15:143,40:212,30:202]
            t1ce = np.load(t1ce)[15:143,40:212,30:202]
            t2 = np.load(t2)[15:143,40:212,30:202]
            
            seg = np.array(nib.load(seg).dataobj)[40:212,30:202,15:143] 
#             print(np.unique(seg), '1') 
            
            seg= np.flipud(rotate(seg, 90, reshape=False)).copy()
#             seg=np.swapaxes(np.swapaxes(seg,1,2),1,0)
            
#             print(np.unique(seg), '2') 
            
            seg[seg==4]=3
            seg=(seg==4)+(seg==2)+(seg==1)+(seg==3) # combine
            
#             print(np.unique(seg), '3') 
            
            flair=standardizer(flair,mod=0) ; t1=standardizer(t1,mod=1); t1ce=standardizer(t1ce,mod=2); t2=standardizer(t2,mod=3)
#             print(type(flair))
            flair=noisy(flair) ; t1=noisy(t1); t1ce=noisy(t1ce); t2=noisy(t2)

            concat_full=np.concatenate([flair,t1,t1ce,t2],axis=0)
            concat_full=torch.tensor(concat_full).transpose(0,1).transpose(1,2).unsqueeze(3)

            rand=np.random.choice([0,1,2,3,4])
#             print(np.shape(seg))
            if rand==0: 
                concat_full=seq.augment_images((np.array(concat_full)))
                seg_orig=torch.tensor( seq.augment_images((np.array(seg).astype(int)))      )
            elif rand==1:
                concat_full=seq2.augment_images((np.array(concat_full)))
                seg_orig=torch.tensor( seq2.augment_images((np.array(seg).astype(int)))      )
            elif rand==2:
                concat_full=seq3.augment_images((np.array(concat_full)))
                seg_orig=torch.tensor( seq3.augment_images((np.array(seg).astype(int)))      )
            else:seg_orig=torch.tensor(seg.astype(int))
            

            flair=torch.tensor(concat_full[:,:,:128,0])
            t1=torch.tensor(concat_full[:,:,128:256,0])
            t1ce=torch.tensor(concat_full[:,:,256:384,0])
            t2=torch.tensor(concat_full[:,:,384:512,0])

            img=np.stack([flair,t1,t1ce,t2])

        
            seg=one_hot(seg_orig) 
            classify=0

            return(img,seg,seg_orig,classify)



class BraTS_FLAIR_val(torch.utils.data.Dataset):  # Validation dataset getter
 

    def __init__(self, csv_file, root_dir, transform=None):  
        self.paths = glob.glob('/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/**')
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx): 
        
        flair=self.paths[idx]+'/'+str(self.paths[idx])[-16:]+'_flairB.npy'  # image files location
        t1=self.paths[idx]+'/'+str(self.paths[idx])[-16:]+'_t1B.npy'
        t1ce=self.paths[idx]+'/'+str(self.paths[idx])[-16:]+'_t1ceB.npy'
        t2=self.paths[idx]+'/'+str(self.paths[idx])[-16:]+'_t2B.npy'        
        seg=self.paths[idx]+'/'+str(self.paths[idx])[-16:]+'_seg.nii'
        
        
        print('Val: ',str(self.paths[idx])[-16:])
        eg_name=str(self.paths[idx])[-16:]
        img,seg,seg_orig,classify=read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg)

        sup_block3=resize(np.array(seg_orig),[172/4, 172/4, 128/4],order=0,mode='constant', preserve_range=True)
        sup_block3=torch.tensor(sup_block3) 
        one_hot_b3=one_hot(sup_block3)#.transpose(0,1) 

        sup_block4=resize(np.array(seg_orig),[172/2, 172/2, 128/2],order=0,mode='constant', preserve_range=True)
        sup_block4=torch.tensor(sup_block4)
        one_hot_b4=one_hot(sup_block4)#.transpose(0,1) 

        sample = {'img': torch.from_numpy(img), 'mask': seg.type(torch.ByteTensor),'seg_orig':seg_orig.type(torch.ByteTensor),'eg_name':eg_name,'classify':classify,
        'sup_block3':sup_block3,'one_hot_b3':one_hot_b3,'sup_block4':sup_block4,'one_hot_b4':one_hot_b4}
        
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

        flair=self.brats[idx]+'/'+self.brats[idx][55:]+'_flairB.npy'  # image files location
        t1=self.brats[idx]+'/'+self.brats[idx][55:]+'_t1B.npy'
        t1ce=self.brats[idx]+'/'+self.brats[idx][55:]+'_t1ceB.npy'
        t2=self.brats[idx]+'/'+self.brats[idx][55:]+'_t2B.npy'
        seg=self.brats[idx]+'/'+self.brats[idx][55:]+'_seg.nii'

        print('Train: ',self.brats[idx][55:])
        eg_name=self.brats[idx][55:]
        

        img,seg,seg_orig,classify= read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg)

        sup_block3=resize(np.array(seg_orig),[172/4, 172/4, 128/4],order=0,mode='constant', preserve_range=True)
        sup_block3=torch.tensor(sup_block3) 
        one_hot_b3=one_hot(sup_block3)#.transpose(0,1)#.squeeze(0) 

        sup_block4=resize(np.array(seg_orig),[172/2, 172/2, 128/2],order=0,mode='constant', preserve_range=True)
        sup_block4=torch.tensor(sup_block4)
        one_hot_b4=one_hot(sup_block4)#.transpose(0,1)#.squeeze(0) 
        # print(sup_block3.size(),one_hot_b3.size())
        sample = {'img': torch.from_numpy(img), 'mask': seg.type(torch.ByteTensor),'seg_orig':seg_orig.type(torch.ByteTensor),'eg_name':eg_name,'classify':classify,
        'sup_block3':sup_block3,'one_hot_b3':one_hot_b3,'sup_block4':sup_block4,'one_hot_b4':one_hot_b4}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

 
 
brats=pd.read_csv('/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv')
hgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/HGG/'
lgg_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/LGG/'
csv_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive3/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/'

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
        try:
            x = torch.cat((x, prev_feature_map), dim=1)
        except:
            x = torch.cat((padder(x), prev_feature_map), dim=1)
            
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

        self.down_block1 = UNet_down_block(4, 62, False)
        self.down_block2 = UNet_down_block(62, 124, True)
        self.down_block3 = UNet_down_block(124, 248, True)
        self.down_block4 = UNet_down_block(248, big, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)

        self.mid_conv1 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(big)
        self.mid_conv2 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(big)
        self.mid_conv3 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(big)

        self.up_block1 = UNet_up_block(big, big, 248)
        self.up_block2 = UNet_up_block(248, 248, 124)
        self.up_block3 = UNet_up_block(124, 124, 62)
        self.up_block4 = UNet_up_block(62, 62, 8)


        self.last_conv1 = torch.nn.Conv3d(12, 8, 3, padding=1) # 12->8 no input concat
        self.last_bn = my_norm(8)
        self.last_conv2 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn2= my_norm(4)
        self.relu = my_activation()
        self.last_conv3 = torch.nn.Conv3d(4, 2, 3, padding=1)#made 4-> 2


        self.trans=torch.nn.ConvTranspose3d(8,8,2,2,padding=0)
        self.convt = torch.nn.Conv3d(8, 8, 3, padding=1)
        self.bnt= my_norm(8)

        self.dsb3 = torch.nn.Conv3d(62, 2, 3, padding=1)#made 4-> 2
        self.dsb4 = torch.nn.Conv3d(8, 2, 3, padding=1)#made 4-> 2
        
        # self.class1 = torch.nn.Conv3d(12, 6, 3, padding=1)
        # self.classbn=my_norm(6)
        # self.class2 = torch.nn.Conv3d(6, 3, 3, padding=1)
        # self.avg=torch.nn.AdaptiveMaxPool3d((1,1,128))

         
    def forward(self, x):
        # print('input unet',x.size())
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
        
        x = self.up_block1(self.x4, self.xmid)
        # print("BlockU 1 shape:",x.size())
        x = self.up_block2(self.x3, x)
        # print("BlockU 2 shape:",x.size())
        
        b3 = self.up_block3(self.x2, x)
        # print("BlockU 3 shape:",b3.size())
 
        b4 = self.up_block4(self.x1, b3) #bs,8,86,86,64
        # print("BlockU 4 shape:",b4.size())
        
        x=  self.trans(b4)   #bs,8,172,172,128  #kernal size is 2
        x = self.bnt(self.relu(self.convt(x)))
        
        x = torch.cat((x, inputs), dim=1) # #bs,12,172,172,128 concat with input  * show x not inps
        
        # classify=self.classbn(self.relu(self.class1(x)))
        # classify=self.class2(classify) #bs,3,172,172,128 
        # pool=self.avg(classify).squeeze(2).squeeze(2)# bs,3,1,1,128->bs,3,128 

        
        
        x = self.last_bn(self.relu(self.last_conv1(x)))
        x = self.last_bn2(self.relu(self.last_conv2(x)))
        x = self.last_conv3(x)
        
        b3=self.dsb3(b3) # supervision
        b4=self.dsb4(b4)
        # print('unet output',x.size())

        return(b3,b4,x)




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
     
    
    return score1,score2



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

# def vis(soft_mask,batch_seg_orig,vis_count,mode='train'):
#     arg=torch.argmax(soft_mask,0)
#     if mode=='train': name='train' 
#     else: name= 'val'
#     for k in range(70,95):
#         writer.add_image(name+' Output', soft_mask[0,:,:,k], vis_count+k)
#         writer.add_image(name+' Target',batch_seg_orig[0,:,:,k] , vis_count+k)


def square_dice(a):
    q=10*a
    q=q**2
    q=q/10
    return(q)
L=np.array([]);Lc=np.array([]);Ld=np.array([]);Lv=np.array([]);Lvd=np.array([]);Lvc=np.array([]);L_r=np.array([]);Ldc=np.array([]);Lvdc=np.array([])


model = UNet().cuda()
# model.load_state_dict(torch.load('/home/Drive3/rahul/unetclass_ifplusreg-14.15|391')['state_dict'])

# model.to(device)
# model = torch.nn.DataParallel(model, device_ids=[ 1,2])
hotlist={}
biglist={}
def train(): 
    global Lc,Lvc,Lvdc, Lc,Ldc,L,Lv
    forward_times=counter_t=vis_count=0

    dataset = BraTS_FLAIR(csv_dir,hgg_dir,transform=None,train_size=train_size)
    dataset_val=BraTS_FLAIR_val(csv_dir,hgg_dir)    #val
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    val_loader=DataLoader(dataset_val,batch_size_val,shuffle=True, num_workers=2) 
    loaders={'train':data_loader,'val': val_loader}

 
    top_models = [(0, 10000)] * 5 # (epoch,loss)
    worst_val = 10000 # init val save loop

    loss_fn = torch.nn.CrossEntropyLoss()  
    soft_max=torch.nn.Softmax(dim=1)
    # bce_classify = torch.nn.BCELoss() 
    # sigmoid=torch.nn.Sigmoid()

    
    opt = torch.optim.Adam(model.parameters(), lr=init_lr,weight_decay=weight_decay)
    scheduler = StepLR(opt, step_size=opt_step_size, gamma=opt_gamma)  #**
    # multisteplr 0.5,

    opt.zero_grad()

    for epoch in range(epochs):
        for e in loaders:
            if e=='train': counter_t+=1; model.train() ; grad=True #
            else: model.eval() ; grad=False


            with torch.set_grad_enabled(grad):    
                for idx, batch_data in enumerate(loaders[e]):
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

                    # print(sup_block3.size(),one_hot_b3.size())
                    

#                     print('b4',sup_block4.size(),one_hot_b4.size())
                    
                    
                    b3,b4,pred_mask = model(batch_input)
                    if e=='train':forward_times += 1

                    ce = loss_fn(pred_mask, batch_seg_orig.long())
                    soft_mask=soft_max(pred_mask)
                    dice=dice_loss(soft_mask, batch_gt_mask)

                    a,b=dice_loss_classes(soft_mask, batch_gt_mask)
                    
                    
                            
                    lossnet = ce+(a+b)/2 # +dice
                    
                    
                    
                    # print('lb3',b3.size(),sup_block3.size())
                    ceb3 = loss_fn(b3, sup_block3.long())
                    soft_mask_b3=soft_max(b3)
                    a1,b1=dice_loss_classes(soft_mask_b3, one_hot_b3.float())
                    lossb3=ceb3+(a1+b1)/2
                    
                    ceb4 = loss_fn(b4, sup_block4.long())
                    soft_mask_b4=soft_max(b4)
                    a2,b2=dice_loss_classes(soft_mask_b4, one_hot_b4.float())
                    lossb4=ceb4+(a2+b2)/2
                    

                    # loss_classify=bce_classify(sigmoid(classify),classify_tars.float())


                    loss=lossnet  +  alpha*lossb3 +  beta*lossb4 #+ thetha*loss_classify

                    print('Dice Losses: ',a.item(),b.item())
                    if e=='train': 
                        if epoch>epoch_magnifiy:
                            if b>0.5 or a>0.5 :
                                loss=3*loss
                                print('Magnified')

                        if epoch>epoch_trouble:
                            if loss>1.5*loss_moving_avg:
                                loss=3*loss
                                print('Trouble')
                        
                        

                        loss.backward()
                        
                        if forward_times == grad_accu_times:
                            Lc=np.concatenate((np.array([ce.item()]), Lc)); cross_moving_avg=np.mean(Lc)
                            Ldc=np.concatenate((np.array([b.item()]), Ldc)); dice1=np.mean(Ldc)
                        
                            L=np.concatenate((np.array([loss.item()]), L)); loss_moving_avg=np.mean(L)
                            print('Epoch: ', epoch+1, ' Batch: ',idx+1,' lr: ',scheduler.get_lr()[-1], ' Dice1: ', dice1, ' CE: ',cross_moving_avg,' Loss :' ,loss_moving_avg)

                            opt.step()
                            opt.zero_grad()
                            forward_times = 0
                            print('\nUpdate weights ... \n')

                        writer.add_scalar('Total Train Loss', loss.item(), counter_t)
                        writer.add_scalar('Target Train Loss', lossnet.item(), counter_t)
                        writer.add_scalar('Train CE', ce.item(), counter_t)
                        # writer.add_scalar('Train Dice', dice.item() , counter_t)
                        writer.add_scalar('D1', a.item(), counter_t)
                        writer.add_scalar('D2', b.item(), counter_t)
                        # writer.add_scalar('D3', c.item() , counter_t)

                        # writer.add_scalar('D4', d.item() , counter_t)
                        writer.add_scalar('lossb3', lossb3.item() , counter_t)
                        writer.add_scalar('lossb4', lossb4.item() , counter_t)
                        # writer.add_scalar('loss_classify', loss_classify.item() , counter_t)
                        writer.add_scalar('Lr', scheduler.get_lr()[-1] , counter_t)
                        torch.cuda.empty_cache()
                        if epoch>log_hot:
                            if b>0.25 or a>0.25 or eg_name[0] in hotlist.keys(): #hotlist,start after N epochs,keep tracking once added
                                print('hotlist culprit')
                                if eg_name[0] not in hotlist.keys():
                                    hotlist[eg_name[0]]=[ np.array(ce.item()) , np.array(b.item()) ]
                                else:
                                    hotlist[eg_name[0]][0]=np.concatenate((hotlist[eg_name[0]][0], np.array(ce.item())))
                                    hotlist[eg_name[0]][1]=np.concatenate((hotlist[eg_name[0]][1], np.array(b.item())))

                                    # hotlist[eg_name[0]][2].append(c.item())
                                    # hotlist[eg_name[0]][3].append(d.item())
                        if epoch>log_epoch:
                            if epoch%20==0:
                                print('biglist updated')

                                if eg_name[0] not in biglist.keys():
                                    biglist[eg_name[0]]=[ np.array(ce.item()) , np.array(b.item()) ]
                                else:
                                    biglist[eg_name[0]][0]=np.concatenate((hotlist[eg_name[0]][0], np.array(ce.item())))
                                    biglist[eg_name[0]][1]=np.concatenate((hotlist[eg_name[0]][1], np.array(b.item())))
                                    # biglist[eg_name[0]][2].append(c.item())
                                    # biglist[eg_name[0]][3].append(d.item())

                        
                        # vis(soft_mask,batch_seg_orig,vis_count,mode='train')
                        # vis_count+=25 # += no of images in vis loop
                        scheduler.step()


                    else:

                        Lv=np.concatenate((np.array([loss.item()]), Lv)) ; lv_avg=np.mean(Lv)
                        Lvc=np.concatenate((np.array([ce.item()]), Lvc)) ;lvc_avg=np.mean(Lvc)
                        Lvdc=np.concatenate((np.array([b.item()]), Lvdc))


                        writer.add_scalar('Val Loss', loss.item(), counter_t)
                        writer.add_scalar('Val CE', ce.item(), counter_t)
                        # writer.add_scalar('Val Dice', dice.item(), counter_t)
                        writer.add_scalar('D1v', a.item(), counter_t)
                        writer.add_scalar('D2v', b.item(), counter_t)
                        # writer.add_scalar('D3v', c.item() , counter_t)
                        # writer.add_scalar('D4v', d.item() , counter_t)
                        # vis(soft_mask,batch_seg_orig,vis_count,mode='val')
                        print(save_initial)

                        print('Validation total Loss::::::::::', round(loss.item(),3) )
                        del batch_input, batch_gt_mask, batch_seg_orig, pred_mask

                        print('current n worst val: ',round(loss.item(),2),worst_val)

                        if epoch>120 and epoch%20==0: # save every 15- for logs- confilicting with down

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
                                
                        break
print('here')
train()
        

    
