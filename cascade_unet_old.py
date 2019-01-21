# added the wierd ifs, no output augs
# weights reg, dice*3
import os

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


import torch
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import imgaug as ia
from imgaug import augmenters as iaa



from torch.autograd import Variable
from torch.utils.data import DataLoader
# from Unet import UNet
# from carvana_dataset import CarvanaDataset
import torch.nn.functional as F
import numpy as np

from scipy import ndimage as nd
import time
import torch as torch
from tensorboardX import SummaryWriter
writer = SummaryWriter()

brats=pd.read_csv('/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv')
 

hgg_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/HGG/'
lgg_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/LGG/'
csv_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/'

def twod_one_hot(targets):    # one hot encodes 3D block to 5 channels with each channel as only one class
    targets_extend=targets.clone().long() # long needed
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.FloatTensor(targets_extend.size(0), 5, targets_extend.size(2), targets_extend.size(3)).zero_() # add zero axis
    one_hot.scatter_(1, targets_extend, 1) # scatter
    return one_hot

class BraTS_FLAIR_val(torch.utils.data.Dataset):  # Validation dataset getter
 

    def __init__(self, csv_file, root_dir, transform=None,val=False):  
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
      
        self.brats = pd.read_csv(csv_file)
        
        self.brats=self.brats.iloc[1:,:]
        self.root_dir = root_dir
        self.transform = transform
        self.val=val

    def __len__(self):
        return len(self.brats)

    def __getitem__(self, idx): 
        
        idx=np.random.randint(0,9)
        
        def read_img( flair,t1,t1ce,t2,seg):
        ####################################
        ###### regular processing###########
        ####################################
            flair = nib.load(flair)   # read image
            t1 = nib.load(t1)
            t1ce = nib.load(t1ce)
            t2 = nib.load(t2)
            seg = nib.load(seg)

            flair = np.array(flair.dataobj) # convert to np with .dataobj
            t1 = np.array(t1.dataobj)
            t1ce= np.array(t1ce.dataobj)
            t2 = np.array(t2.dataobj)
            seg = np.array(seg.dataobj)

            def standardizer(img):
                mean=np.mean(img)
                std=np.std(img)
                img=img-mean
                img=img/std
                
                return(img)
            flair=standardizer(flair)
            t1=standardizer(t1)
            t1ce=standardizer(t1ce)
            t2=standardizer(t2)

            zoom=0.41

            flair = nd.interpolation.zoom(flair, zoom=zoom)  # downsample only inputs
            t1 = nd.interpolation.zoom(t1, zoom=zoom)
            t1ce = nd.interpolation.zoom(t1ce, zoom=zoom)
            t2 = nd.interpolation.zoom(t2, zoom=zoom)
            seg = nd.interpolation.zoom(seg, zoom=zoom)
            seg=(seg==4)+(seg==2)+(seg==1)+(seg==3) # ** entire tumor=1

            img=np.stack([flair,t1,t1ce,t2])


            ####################################################
            ###### concat only inputs and data augment them ###########
            #####################################################

            concat=np.concatenate([flair,t1,t1ce,t2],axis=2) 


      
            concat=concat[1:-1,1:-1,:]  # reshape for /8 specific to zoom=0.41
            seg=seg[1:-1,1:-1,:]

            concat_full=np.concatenate([concat,seg],axis=2)



            #             images_aug=self.seq_seg.augment_images(concat_full)
            images_aug=concat_full

            flair=images_aug[:,:,:64]
            t1=images_aug[:,:,64:128]
            t1ce=images_aug[:,:,128:192]
            t2=images_aug[:,:,192:256]
            seg=images_aug[:,:,256:]




            seg_orig=np.clip(seg.copy(),0,2)      # <- size=[bs,96,96,64] class number as pixel value

            #one-hot encode
            seg=torch.from_numpy(seg) # convert to torch
            ten=twod_one_hot(seg)
            ten.transpose_(0,1)


            seg=ten

            return(img,seg,seg_orig)

        
        paths = glob.glob('/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/**')
            


        flair=paths[idx]+'/'+str(paths[idx])[-16:]+'_flair.nii'  # image files location
        t1=paths[idx]+'/'+str(paths[idx])[-16:]+'_t1.nii'
        t1ce=paths[idx]+'/'+str(paths[idx])[-16:]+'_t1ce.nii'
        t2=paths[idx]+'/'+str(paths[idx])[-16:]+'_t2.nii'

        seg=paths[idx]+'/'+str(paths[idx])[-16:]+'_seg.nii'

        img,seg,seg_orig=read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg)

        img=torch.from_numpy(img)
        seg_orig=torch.from_numpy(seg_orig)


        sample = {'img': img, 'mask': seg,'seg_orig':seg_orig}
       
        if self.transform:
            sample = self.transform(sample)

        return sample



class BraTS_FLAIR(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, transform=None,val=False): # error in i=0,77 missing
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
 

        self.brats = pd.read_csv(csv_file)
        self.brats=self.brats.iloc[1:,:]
        self.root_dir = root_dir
        self.transform = transform
        self.val=val
        sometimes=lambda aug: iaa.Sometimes(0.5, aug)
        self.seq=iaa.Sequential([

            iaa.Sometimes(0.2,iaa.OneOf(iaa.GaussianBlur(sigma=(1, 1.3)),iaa.AverageBlur(k=((5, 5), (1, 3))))) , # gauss OR avg blue 

            iaa.Sometimes(0.2,iaa.Dropout((0.03, 0.08))) ])

 #           iaa.Sometimes(0.2,iaa.OneOf(iaa.MultiplyElementwise((0.8, 1.3)),iaa.ElasticTransformation(alpha=(1, 1.34), sigma=0.1)))   
 #                         ])
#         self.seq_seg =iaa.Sequential([ 
            
#             ])
                
        
#         self.seq_seg =iaa.Sequential([ 
#             self.sometimes(iaa.SomeOf((0, 2),[iaa.Crop(px=(0, 14)), 
#             iaa.Fliplr(0.5),
#             iaa.Flipud(0.5)]))
#                 ])
    def __len__(self):
        return len(self.brats)
    
    
    def noisy(self,image): # only for inputs
        #choice=np.random.choice([0,0,0,0,0,0,0,0,1,1]) # add noise to 20% of the images
        choice=1
        if choice==1:
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            noisy = nd.gaussian_filter(noisy, sigma=(0.7, 0.7, 0.7), order=0)
            
            return noisy
        else:
            return image
    


        
    def read_img(self, flair,t1,t1ce,t2,seg):
            ####################################
            ###### regular processing###########
            ####################################
            flair = nib.load(flair)   # read image
            t1 = nib.load(t1)
            t1ce = nib.load(t1ce)
            t2 = nib.load(t2)
            seg = nib.load(seg)
            
            flair = np.array(flair.dataobj) # convert to np with .dataobj
            t1 = np.array(t1.dataobj)
            t1ce= np.array(t1ce.dataobj)
            t2 = np.array(t2.dataobj)
            seg = np.array(seg.dataobj)
            
            def standardizer(img):
                mean=np.mean(img)
                std=np.std(img)
                img=img-mean
                img=img/std
                
                return(img)
            flair=standardizer(flair)
            t1=standardizer(t1)
            t1ce=standardizer(t1ce)
            t2=standardizer(t2)

            zoom=0.41

            flair = nd.interpolation.zoom(flair, zoom=zoom)  # downsample only inputs
            t1 = nd.interpolation.zoom(t1, zoom=zoom)
            t1ce = nd.interpolation.zoom(t1ce, zoom=zoom)
            t2 = nd.interpolation.zoom(t2, zoom=zoom)
            seg = nd.interpolation.zoom(seg, zoom=zoom)
            seg=(seg==4)+(seg==2)+(seg==1)+(seg==3) # ** entire tumor=1
            
            img=np.stack([flair,t1,t1ce,t2])
            
            
            ####################################################
            ###### concat only inputs and data augment them ###########
            #####################################################
            
            concat=np.concatenate([flair,t1,t1ce,t2],axis=2) 
            
            
            concat=self.noisy(concat) # add noise with prob p
            concat = self.seq.augment_images(concat) # add augmentations with related probs
            
            concat=concat[1:-1,1:-1,:]  # reshape for /8 specific to zoom=0.41
            seg=seg[1:-1,1:-1,:]
            
            concat_full=np.concatenate([concat,seg],axis=2)
            
            
            
#             images_aug=self.seq_seg.augment_images(concat_full)
            images_aug=concat_full
            
            flair=images_aug[:,:,:64]
            t1=images_aug[:,:,64:128]
            t1ce=images_aug[:,:,128:192]
            t2=images_aug[:,:,192:256]
            seg=images_aug[:,:,256:]

            
            

            seg_orig=np.clip(seg.copy(),0,2)      # <- size=[bs,96,96,64] class number as pixel value
            
            #one-hot encode
            seg=torch.from_numpy(seg) # convert to torch
            ten=twod_one_hot(seg)
            ten.transpose_(0,1)


            seg=ten
            
            return(img,seg,seg_orig)
    

    def __getitem__(self, idx): 

        if idx==0:
            idx=np.random.randint(100)

        try:
            folder=os.path.join(hgg_dir,self.brats.iloc[idx, 0])# patient folder   # create if loop for hgg/lgg
        except:
            folder=os.path.join(lgg_dir,self.brats.iloc[idx, 0])
            
        flair=folder+'/'+str(self.brats.iloc[idx, 0])+'_flair.nii.gz'  # check zip files
        t1=folder+'/'+str(self.brats.iloc[idx, 0])+'_t1.nii.gz'
        t1ce=folder+'/'+str(self.brats.iloc[idx, 0])+'_t1ce.nii.gz'
        t2=folder+'/'+str(self.brats.iloc[idx, 0])+'_t2.nii.gz'
        seg=folder+'/'+str(self.brats.iloc[idx, 0])+'_seg.nii.gz'

        try:    # unzip if not already unzipped
            gunzip(flair)
        except:
            pass
        try:
            gunzip(t1)
        except:
            pass
        try:
            gunzip(t1ce)
        except:
            pass
        try:
            gunzip(t2)
        except:
            pass
        try:
            gunzip(seg)
        except:
            pass
        

        flair=folder+'/'+str(self.brats.iloc[idx, 0])+'_flair.nii'  # image files location
        t1=folder+'/'+str(self.brats.iloc[idx, 0])+'_t1.nii'
        t1ce=folder+'/'+str(self.brats.iloc[idx, 0])+'_t1ce.nii'
        t2=folder+'/'+str(self.brats.iloc[idx, 0])+'_t2.nii'
        seg=folder+'/'+str(self.brats.iloc[idx, 0])+'_seg.nii'

        img,seg,seg_orig= self.read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg)
        

        
        
        
        sample = {'img': torch.from_numpy(img), 'mask': seg,'seg_orig':torch.from_numpy(seg_orig)}
        
        if self.transform:
            sample = self.transform(sample)

        return sample













class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.max_pool = torch.nn.MaxPool3d(2, 2)
        self.relu = torch.nn.ELU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        residual=x.clone()
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x=x+residual
        return x

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
#         self.up_sampling = torch.nn.functional.interpolate(scale_factor=2, mode='trilinear')
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.relu = torch.nn.ELU()

    def forward(self, prev_feature_map, x):
        
        x = torch.nn.functional.interpolate(x,scale_factor=2, mode='trilinear')
        
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        residual=x.clone()
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x=x+residual
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(4, 24, False)
        self.down_block2 = UNet_down_block(24, 72, True)
        self.down_block3 = UNet_down_block(72, 148, True)
        self.down_block4 = UNet_down_block(148, 224, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)


        
        self.mid_conv1 = torch.nn.Conv3d(224, 224, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(224)
        self.mid_conv2 = torch.nn.Conv3d(224, 224, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(224)
        self.mid_conv3 = torch.nn.Conv3d(224, 224, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(224)

        
        self.up_block1 = UNet_up_block(224, 224, 148)
        self.up_block2 = UNet_up_block(148, 148, 72)
        self.up_block3 = UNet_up_block(72, 72, 24)
        self.up_block4 = UNet_up_block(24, 24, 8)

        self.last_conv1 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm3d(4)
        self.last_conv2 = torch.nn.Conv3d(4, 1, 1, padding=0)
        self.relu = torch.nn.ELU()
        self.last_conv3 = torch.nn.Conv3d(1, 1, 1, padding=0)
        self.relu = torch.nn.ELU()
        
        self.conv1f=torch.nn.Conv2d(1, 2, 3,padding=1)
        self.conv2f=torch.nn.Conv2d(2,2, 3,padding=1)
        self.conv3f=torch.nn.Conv2d(2, 2, 3,padding=1)

    def forward(self, x):
#         print('input unet',x.size())
        self.x1 = self.down_block1(x)
#         print("Block 1 shape:",self.x1.size())
        self.x2 = self.down_block2(self.x1)
        if self.x2.size()[2]==49:                                         ###*********************************** ifffff        if self.x2.size()[2]==49:
            self.x2=self.x2[:,:,1:,1:,:]

            
#         print("Block 2 shape:",self.x2.size())
        self.x3 = self.down_block3(self.x2)
#         print("Block 3 shape:",self.x3.size())
        
            
        self.x4 = self.down_block4(self.x3)
#         print("Block 4 shape:",self.x4.size())
        
        
        self.xmid=self.max_pool(self.x4)
        self.xmid = self.relu(self.bn1(self.mid_conv1(self.xmid)))
        self.xmid = self.relu(self.bn2(self.mid_conv2(self.xmid)))
        self.xmid = self.relu(self.bn3(self.mid_conv3(self.xmid)))
#         print("Block Mid shape:",self.xmid.size())
        
        
        
        x = self.up_block1(self.x4, self.xmid)
#         print("BlockU 1 shape:",x.size())
        x = self.up_block2(self.x3, x)
#         print("BlockU 2 shape:",x.size())
        
        x = self.up_block3(self.x2, x)
#         print("BlockU 3 shape:",x.size())

        if self.x1.size()[2]==98:                     ###*********************************** ifffff
            self.x1=self.x1[:,:,1:-1,1:-1,:]
            #print('chan98',self.x1.size())

        x = self.up_block4(self.x1, x)
#         print("BlockU 4 shape:",x.size())
        
        
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)  # of size [batch_size,1,h,w,depth] or [bs, modalities(1) ,96 ,96 , 64]
        
        x=x.view(batch_size,1,-1,64)
#         x=x.squeeze(1)  
#         print('input convf',x.size())
        conv=self.relu(self.conv1f(x))
        conv=self.relu(self.conv2f(conv))
        conv=self.conv3f(conv)
        
        
        try:
            conv=conv.view(batch_size,2,96,96,64)
        except:
            conv=conv.view(batch_size_val,2,96,96,64)
#         print('unet output',conv.size())
        
        return(conv)

if __name__ == '__main__':
    net = UNet().cuda()
    print(net)


loss_weights=torch.tensor([0.0000171209,0.001027749229+0.0003360215054+0.00487804878+0.001071811361],requires_grad=False).cuda()

def dice_loss(inpu, target):  # [bs,2,96,96,64]
    smooth = 1.
    def calc(iflat,tflat):
        intersection = (iflat * tflat).sum() 
        
        return (((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth)))
    
    
    a=0
    b=0
    c=0
    d=0
    e=0
    
    for k in range(inpu.size()[0]):
        ip=inpu[k,0,:,:,:].view(-1)
        tar=target[k,0,:,:,:].view(-1)
        a+=calc(ip,tar)
        
        ip=inpu[k,1,:,:,:].view(-1)
        tar=target[k,1,:,:,:].view(-1)
        b+=calc(ip,tar)
        
    
    raw_scores=(loss_weights[0]*a + loss_weights[1]*b)/2*(k+1)
    
    return 1.0 - raw_scores
    



batch_size = 2
batch_size_val=2
def train():
    counter=0
    L=[]
    Lc=[]
    Ld=[]
    Lv=[]
    Lvd=[]
    Lvc=[]
    L_r=[]
    grad_accu_times = 4
    init_lr = 0.01

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = BraTS_FLAIR(csv_dir,hgg_dir)
    dataset_val=BraTS_FLAIR_val(csv_dir,hgg_dir)    #val


    model = UNet().cuda()
    torch.save(model.state_dict(), "recent.pt")
    top_ten_models = [(0, 0)] * 10

    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)  # appies softmax then computes loss
    soft_max=torch.nn.Softmax(dim=1)
    
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    opt.zero_grad()

    epoch = 0
    forward_times = 0
    best_loss=10000
 
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    for epoch in range(200):
        
        if epoch >= 0:
            val_loader=DataLoader(dataset_val,batch_size_val,shuffle=True, num_workers=2) # val
            count=0
            model.eval()
            
            print('in val')
            for idx, batch_data in enumerate(val_loader):
                batch_data['img']=batch_data['img'].float()
                batch_data['mask']=batch_data['mask'].float()
                batch_data['seg_orig']=batch_data['seg_orig'].float()
                
                
                batch_input = Variable(batch_data['img'],volatile=True).cuda()
                batch_gt_mask = Variable(batch_data['mask'],volatile=True).cuda()
                batch_seg_orig=Variable(batch_data['seg_orig'],volatile=True).cuda()
                pred_mask = model(batch_input)
                
                loss = loss_fn(pred_mask, batch_seg_orig.long())
                print(loss[0], '???????????? Validation Loss for Cross entropy ?????????????????????????')
                
                soft_mask=soft_max(pred_mask)
                
                dice=dice_loss(soft_mask, batch_gt_mask)
                print('the validaTION DICE     >>>>>>>>>>>>>>>' ,dice[0],'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            
                loss_val=loss+dice
                vd=dice.item()
                vc=loss.item()
                v=loss_val.item()
                print('Validation total Loss::::::::::', loss_val[0])
                
                
                del batch_input, batch_gt_mask, batch_seg_orig, pred_mask
                print('out')
                count+=1
                if count>1:
                    in_val=False
                    Lv.append(v)
                    Lvd.append(vd)
                    Lvc.append(vc)
                    break
        
        if epoch<=60:
            lr = init_lr * (0.5 ** (epoch // 15))
            fin_lr=lr
            L_r.append(fin_lr)
        elif 60<=epoch<=400:
            lr = 2*fin_lr* (0.7 ** ((epoch-60) // 50))
            fin2=lr
            L_r.append(fin2)
        elif 500<=epoch<=900:
            lr = fin2* (0.75 ** ((epoch-500) // 60))
            L_r.append(lr)



        for param_group in opt.param_groups:
            param_group['lr'] = lr

    
        for idx, batch_data in enumerate(data_loader):
            counter+=1
            model.train()
            batch_data['img']=batch_data['img'].float()
            batch_data['mask']=batch_data['mask'].float()
            batch_data['seg_orig']=batch_data['seg_orig'].float()

            batch_input = Variable(batch_data['img']).cuda()
            batch_gt_mask = Variable(batch_data['mask']).cuda()
            batch_seg_orig=Variable(batch_data['seg_orig']).cuda()
#             print('input img and mask sizes: ', batch_input.size(), batch_gt_mask.size())

            
            pred_mask = model(batch_input)


            forward_times += 1

            

            loss = loss_fn(pred_mask, batch_seg_orig.long())
            Lc.append(loss.cpu().data.numpy())
            cross_moving_avg=sum(Lc)/len(Lc)
            
            print('cross entropy loss mving average is  .....          ',cross_moving_avg)


            soft_mask=soft_max(pred_mask)
            loss += dice_loss(soft_mask, batch_gt_mask) # dice
            
            dice=dice_loss(soft_mask, batch_gt_mask)
            Ld.append(dice.cpu().data.numpy())
            dice_moving_avg=sum(Ld)/len(Ld)
            
            print('dice entropy loss mving average is  .....        ...',dice_moving_avg,',...')



            L.append(loss.cpu().data.numpy())
            loss_moving_avg=sum(L)/len(L)
            #print('total loss',loss)
            
            loss.backward()
            loss_moving_avg=sum(L)/len(L)
            
#             return(pred_mask,batch_gt_mask,soft_mask)
            writer.add_scalar('Total Loss', loss_moving_avg, counter)
            
            print('Epoch {:>3} | Batch {:>5} | lr {:>1.5f} | Loss {:>1.5f} '.format(epoch+1, idx+1, lr, loss.cpu().data.numpy()))
            writer.add_scalar('CE', cross_moving_avg, counter)
            writer.add_scalar('Dice', dice_moving_avg, counter)

            #print('loss',epoch, loss)
            print('loss, epoch, moving average',loss,epoch,loss_moving_avg)
            if forward_times == grad_accu_times:
                opt.step()
                opt.zero_grad()
                forward_times = 0
                print('\nUpdate weights ... \n')

            if (epoch+1) % 2 == 0:
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'moving loss':L,
                    'dice':Ld,
                    'val':Lv,
                    'valc':Lvc,
                    'vald':Lvd,
                    'cross el':Lc,
                    'state_dict': model.state_dict(),
                    'optimizer' : opt.state_dict(),
                }
                torch.save(checkpoint, 'unet1024v-{}'.format(epoch+1))
#             del data_loader


            least_acc = top_ten_models[0][0]    
            if loss < best_loss:              #option: epoch_acc > least_acc
                if epoch > 9:
                    to_be_deleted = "models"+str(top_ten_models[0][1])+'.pt'
                    os.remove(to_be_deleted)
                top_ten_models.pop(0)
                #temp_in = copy.deepcopy(model.state_dict())
                # model_file_name = "models/" + str(epoch) + ".pt"
                model_file_name = "models"+str(epoch)+".pt"
                torch.save(model.state_dict(), model_file_name)
                top_ten_models.append((loss, epoch))
                #save this list->dictionary as pt
                top_ten_models = sorted(top_ten_models, key = lambda x:x[0])
                best_loss=top_ten_models[0][0]
                           
train()
#loss_weights=torch.tensor([650*0.0000171209,136*0.001027749229,100*0.0003360215054,90*0.00487804878,136*0.001071811361],requires_grad=True).cuda()
#train()

soft_max=torch.nn.Softmax(dim=1)

soft_mask=soft_max(pred_mask)
a=soft_mask.argmax(1)


