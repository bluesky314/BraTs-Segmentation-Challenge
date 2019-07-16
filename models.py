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

# Hyperparamaters
dropout_toggel,dropblock_toggle=True, False   
dropblock_blocks=5
dropblock_mid=9
dropout_amount=0.3
leaky_relu=True;relu=False;elu=False 
batch_norm=False 


class my_norm(torch.nn.Module):
    ''' Normalization Module '''
    def __init__(self, channels):
        super(my_norm,self).__init__()
        self.norm=torch.nn.BatchNorm3d(channels) if batch_norm==True else torch.nn.InstanceNorm3d(channels, affine=True)
    def forward(self,x): return(self.norm(x))


class my_activation(torch.nn.Module):
     ''' Activation Function'''

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



class UNet_base(torch.nn.Module):  
    ''' Whole Tumor Unet'''
    
    def __init__(self):
        super(UNet_base, self).__init__()

        self.preconv=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb = my_norm(4)
        self.preconv2=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb2 = my_norm(4)
        self.preconv3=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb3 = my_norm(4)
        self.firstconv=torch.nn.Conv3d(4,4,3,stride=2,padding=1)

        self.down_block1 = UNet_down_block(4, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 248, True)
        self.down_block4 = UNet_down_block(248, 524, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)

        self.mid_conv1 = torch.nn.Conv3d(524, 524, 3, padding=1)

        self.mid_conv2 = torch.nn.Conv3d(524, 524, 3, padding=1)

        self.mid_conv3 = torch.nn.Conv3d(524, 524, 3, padding=1)

        
        self.up_block1 = UNet_up_block(524, 524, 248)
        self.up_block2 = UNet_up_block(248, 248, 128)
        self.up_block3 = UNet_up_block(128, 128, 64)
        self.up_block4 = UNet_up_block(64, 64, 8)


        self.last_conv1 = torch.nn.Conv3d(12, 8, 3, padding=1) 
        self.last_bn = my_norm(8)
        self.last_conv2 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn2= my_norm(4)
        self.relu = my_activation()
        self.last_conv3 = torch.nn.Conv3d(4, 2, 3, padding=1)


        self.trans=torch.nn.ConvTranspose3d(8,8,2,2,padding=0)
        self.convt = torch.nn.Conv3d(8, 8, 3, padding=1)
        self.bnt= my_norm(8)

        self.dsb3 = torch.nn.Conv3d(64, 2, 3, padding=1) # Deep Supervision
        self.dsb4 = torch.nn.Conv3d(8, 2, 3, padding=1)

    def forward(self, x):
        inputs = x.clone()

        x = self.preconvb(self.relu(self.preconv(x)))
        x = self.preconvb2(self.relu(self.preconv2(x)))
        x = self.preconvb3(self.relu(self.preconv3(x)))

        x=self.relu(self.firstconv(x))

        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)    
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
    
        self.xmid=self.max_pool(self.x4)
        self.xmid = self.relu(self.mid_conv1(self.xmid))
        residual=self.xmid.clone()
        self.xmid = self.relu(self.mid_conv2(self.xmid))
        self.xmid = self.relu(self.mid_conv3(self.xmid))
        self.xmid = self.xmid + residual
        
        b1 = self.up_block1(self.x4, self.xmid)
        b2 = self.up_block2(self.x3, b1)
        b3 = self.up_block3(self.x2, b2)
        b4 = self.up_block4(self.x1, b3) 
        
        x=  self.trans(b4)  
        x = self.bnt(self.relu(self.convt(x)))
        
        x = torch.cat((x, inputs), dim=1)  

        x = self.last_bn(self.relu(self.last_conv1(x)))
        x = self.last_bn2(self.relu(self.last_conv2(x)))
        x = self.last_conv3(x)
        
        b3=self.dsb3(b3) # Deep supervision
        b4=self.dsb4(b4)


        return(b3,b4,x)
    
class Input_Block(torch.nn.Module): 
    def __init__(self, input_channel, output_channel):
        super(Input_Block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=1)
        self.bn1 = my_norm(output_channel) 
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = my_norm(output_channel)
        self.relu = my_activation()
        if dropout_toggel==True: self.dropout=torch.nn.Dropout3d(dropout_amount)
        if dropblock_toggle==True: self.dropout=LinearScheduler(DropBlock3D(block_size=dropblock_blocks, drop_prob=0.),
                start_value=0.,
                stop_value=0.25,
                nr_steps=5)


    def forward(self, x):
        x = self.dropout(self.bn1(self.relu(self.conv1(x))))
        x = self.dropout(self.bn2(self.relu(self.conv2(x))))
        return x
    
class UNet_Core(torch.nn.Module):       
    ''' Unet for tumor core region'''
    def __init__(self):
        super(UNet_Core, self).__init__()
        
        self.flair_block=Input_Block(1,4)
        self.t1_block=Input_Block(1,4)
        self.t1ce_block=Input_Block(1,4)
        self.t2_block=Input_Block(1,4)
        
        self.preconv=torch.nn.Conv3d(16,16,3,stride=1,padding=1)
        self.preconvb = my_norm(16)
        self.preconv2=torch.nn.Conv3d(16,16,3,stride=1,padding=1)
        self.preconvb2 = my_norm(16)
        self.preconv3=torch.nn.Conv3d(16,16,3,stride=1,padding=1)
        self.preconvb3 = my_norm(16)

        self.firstconv=torch.nn.Conv3d(16,16,3,stride=2,padding=1)
        
        big=624
        s=[82,212,312]
        
        self.down_block1 = UNet_down_block(16, s[0], False)
        self.down_block2 = UNet_down_block(s[0], s[1], True)
        self.down_block3 = UNet_down_block(s[1], s[2], True)
        self.down_block4 = UNet_down_block(s[2], big, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)

        self.mid_conv1 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn1 = torch.nn.InstanceNorm3d(big)
        self.mid_conv2 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn2 = torch.nn.InstanceNorm3d(big)
        self.mid_conv3 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn3 = torch.nn.InstanceNorm3d(big)

        self.up_block1 = UNet_up_block(big, big, s[2])
        self.up_block2 = UNet_up_block(s[2], s[2], s[1])
        self.up_block3 = UNet_up_block(s[1], s[1], s[0])
        self.up_block4 = UNet_up_block(s[0], s[0], 8)
        
        self.trans=torch.nn.ConvTranspose3d(8,8,2,2,padding=0)
        self.convt = torch.nn.Conv3d(8, 8, 3, padding=1)
        self.bnt= my_norm(8)


        self.last_conv1 = torch.nn.Conv3d(12, 8, 3, padding=1)
        self.last_bn = my_norm(8)
        self.last_conv2 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn2= my_norm(4)
        self.relu = my_activation()
        self.last_conv3 = torch.nn.Conv3d(4, 2, 3, padding=1)

        self.dsb3 = torch.nn.Conv3d(s[0], 2, 3, padding=1)
        self.dsb4 = torch.nn.Conv3d(8, 2, 3, padding=1)

    def forward(self, x):
        inputs = x.clone()
    
        f=self.flair_block(x[:,0].unsqueeze(1))
        t=self.t1_block(x[:,1].unsqueeze(1))
        t1=self.t1ce_block(x[:,2].unsqueeze(1))
        t2=self.t2_block(x[:,3].unsqueeze(1))
        
        
        x = torch.cat((f,t,t1,t2), dim=1)

        x = self.preconvb(self.relu(self.preconv(x)))
        x = self.preconvb2(self.relu(self.preconv2(x)))
        x = self.preconvb3(self.relu(self.preconv3(x)))

        x=self.relu(self.firstconv(x))

        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
    
        self.x4 = self.down_block4(self.x3)
    
        self.xmid=self.max_pool(self.x4)
        self.xmid = self.bn1(self.relu(self.mid_conv1(self.xmid)))
        residual=self.xmid.clone()
        self.xmid = self.bn2(self.relu(self.mid_conv2(self.xmid)))
        self.xmid = self.bn3(self.relu(self.mid_conv3(self.xmid)))
        self.xmid = self.xmid + residual
        
        b1 = self.up_block1(self.x4, self.xmid)
        b2 = self.up_block2(self.x3, b1)       
        b3 = self.up_block3(self.x2, b2)
        b4 = self.up_block4(self.x1, b3)
        
        x=  self.trans(b4)   
        x = self.bnt(self.relu(self.convt(x)))
        x = torch.cat((x, inputs), dim=1) 
        x = self.last_bn(self.relu(self.last_conv1(x)))
        x = self.last_bn2(self.relu(self.last_conv2(x)))
        x = self.last_conv3(x)
        
        b3=self.dsb3(b3) 
        b4=self.dsb4(b4)
        return(b3,b4,x)


class UNet_Enchancing(torch.nn.Module):  
    ''' Unet for enhancing tumor'''
    
    def __init__(self):
        super(UNet_Whole, self).__init__()

        self.preconv=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb = my_norm(4)
        self.preconv2=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb2 = my_norm(4)
        self.preconv3=torch.nn.Conv3d(4,4,3,stride=1,padding=1)
        self.preconvb3 = my_norm(4)

        self.firstconv=torch.nn.Conv3d(4,4,3,stride=2,padding=1)
        
        big=624
        s=[82,212,312]
        
        self.down_block1 = UNet_down_block(4, s[0], False)
        self.down_block2 = UNet_down_block(s[0], s[1], True)
        self.down_block3 = UNet_down_block(s[1], s[2], True)
        self.down_block4 = UNet_down_block(s[2], big, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)

        self.mid_conv1 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn1 = torch.nn.InstanceNorm3d(big)
        self.mid_conv2 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn2 = torch.nn.InstanceNorm3d(big)
        self.mid_conv3 = torch.nn.Conv3d(big, big, 3, padding=1)
        self.bn3 = torch.nn.InstanceNorm3d(big)

        self.up_block1 = UNet_up_block(big, big, s[2])
        self.up_block2 = UNet_up_block(s[2], s[2], s[1])
        self.up_block3 = UNet_up_block(s[1], s[1], s[0])
        self.up_block4 = UNet_up_block(s[0], s[0], 8)
        
        self.trans=torch.nn.ConvTranspose3d(8,8,2,2,padding=0)
        self.convt = torch.nn.Conv3d(8, 8, 3, padding=1)
        self.bnt= my_norm(8)


        self.last_conv1 = torch.nn.Conv3d(12, 8, 3, padding=1) 
        self.last_bn = my_norm(8)
        self.last_conv2 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn2= my_norm(4)
        self.relu = my_activation()
        self.last_conv3 = torch.nn.Conv3d(4, 2, 3, padding=1)

        self.dsb3 = torch.nn.Conv3d(s[0], 2, 3, padding=1)
        self.dsb4 = torch.nn.Conv3d(8, 2, 3, padding=1)

    def forward(self, x):
        inputs = x.clone()

        x = self.preconvb(self.relu(self.preconv(x)))
        x = self.preconvb2(self.relu(self.preconv2(x)))
        x = self.preconvb3(self.relu(self.preconv3(x)))
        x=self.relu(self.firstconv(x))

        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
    
        self.xmid=self.max_pool(self.x4)
        self.xmid = self.bn1(self.relu(self.mid_conv1(self.xmid)))
        residual=self.xmid.clone()
        self.xmid = self.bn2(self.relu(self.mid_conv2(self.xmid)))
        self.xmid = self.bn3(self.relu(self.mid_conv3(self.xmid)))
        self.xmid = self.xmid + residual
        
        b1 = self.up_block1(self.x4, self.xmid)
        b2 = self.up_block2(self.x3, b1)
        b3 = self.up_block3(self.x2, b2)
        b4 = self.up_block4(self.x1, b3) 
        
        x =  self.trans(b4)   
        x = self.bnt(self.relu(self.convt(x)))
        x = torch.cat((x, inputs), dim=1)  
        x = self.last_bn(self.relu(self.last_conv1(x)))
        x = self.last_bn2(self.relu(self.last_conv2(x)))
        x = self.last_conv3(x)
        
        b3=self.dsb3(b3) 
        b4=self.dsb4(b4)


        return(b3,b4,x)



