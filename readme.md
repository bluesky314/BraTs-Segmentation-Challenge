In Progress: Some files are old and update will be coming soon.

This repository contains some files from MICCA BraTS segmentation challenge: https://www.med.upenn.edu/sbia/brats2018/data.html

## Table of Contents
1. [Dataset](#dataset)
2. [MRI Background](#mri-background)
    * [MRI Pre-Processing](#mri-pre-processing)
    * [Pulse Sequences](#pulse-sequences)
    * [Segmentation](#segmentation)
3. [High Grade Gliomas](#high-grade-gliomas)
4. [Convolutional Neural Networks](#convolutional-neural-networks)
    * [Model Architecture](#model-architecture)
    * [Training the Model](#training-the-model)  
    * [Patch Selection](#patch-selection)
    * [Results](#results)
5. [Future Directions](#future-directions)


## Dataset
BraTS 2018 utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. All MRI data was provided by the [2018 MICCAI BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/tasks.html), which consists of 210 high-grade glioma cases and 75 low-grade cases. Each patient has four different MRI [modalities](#modalities), each of which is comprised of 155 brain slices, for a total of 620 images per patient. Professional segmentation is provided as ground truth labels for each case. 

The three tumor regions consist of the whole tumor (yellow), the tumor core (red), the enhancing tumor structures(blue). Due to this highly diverse appearance and shape, segmentation of brain tumors in multimodal MRI scans is one of the most challenging tasks in medical image analysis. This intrinsic diversity of gliomas is also portrayed in their imaging features (appearance and shape), and their sub-regions are described by varying intensity profiles spread across multimodal MRI scans, reflecting varying tumor biological properties. 


<div id="container">
    <img src="https://www.med.upenn.edu/sbia/assets/user-content/BRATS_tasks.png?raw=true" width="550" height="250" >
</div>

Different Tumor Classes and Modalities

Shown above are image patches that are annotated witht the different types of classes in the different modalities (top left) and the final labels for the whole dataset (right). The image patches show from left to right: the whole tumor (yellow) visible in T2-FLAIR (Fig.A), the tumor core (red) visible in T2 (Fig.B), the enhancing tumor structures (light blue) visible in T1Gd, surrounding the necrotic components of the core (green) (Fig. C). The segmentations are combined to generate the final labels of the tumor sub-regions.

<img alt="3D rendering produced by T2 MRI scan" src="https://github.com/naldeborgh7575/brain_segmentation/blob/master/images/t29_143.gif" width=250>  
<sub> <b> Figure 2: </b> (Left) Slices are taken axially at 1mm increments, creating a 3-dimensional rendering of the brain. Note that shown above is only one of four pulse sequences used for tumor segmentation. </sub>

## Data pre-processing 
One of the challenges in working with MRI data is dealing with the artifacts produced either by inhomogeneity in the magnetic field or small movements made by the patient during scan time. So, one part of an image might appear lighter or darker when visualized solely because of variations in the magnetic field. The map of these variations is called the bias field. The bias field can cause problems for a classifier as the variations in signal intensity are not due to any anatomical differences or difference in features.

Often times a bias will be present across the resulting scans, which negatively affect the segmentation results.  We notice that out of 285 total patients, 210 are not much affected by bias correction however 75 are greatly affected indicating a split in the dataset's distribution. Bias correcting raises results on these samples substantially. Most papers do not speak much the split in domain and how there exists majority and minority features classes. The paper [Automatic Brain Tumor Segmentation with Domain Transfer] (chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2018_proceedings_shortPapers.pdf) is the only one we found who took this topic seriously and made provisions to adjust for it. While most papers have ignored the topic completely, few papers in passing have mentioned that their models performed very poorly on a handful of samples in their validation. 

<div id="container">
    <img src="https://github.com/naldeborgh7575/brain_segmentation/blob/master/images/n4_correction.png?raw=true" width="300" height="150" >
</div>

We employ SITK's bias correction on all T1,T2 and T1CE images.  which removed the intensity gradient on each scan. Additional image pre-processing requires standardizing the pixel intensities, since MRI intensities are expressed in arbitrary units and may differ significantly between machines used and scan times. We remove as much of background pixels as possible and crop each image to (128,128,172). We normalise by removing the top 2% pixel value and by median as there is alot of high frequency noise in the data. We also rescale each image to (0,1). We found this greatly speeds up learning.

<div id="container">
    <img src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/brain_grids.png?raw=true" width="800">
</div>

<div id="container">
    <img src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/segment.png?raw=true" width="800" >
</div>


## Convolutional Neural Networks

Convolution Neural Networks have proven to be vastly superior to other hand-crafted and learning algorithms for complex feature representation. To this end, we employ a framework based on U-Net  structure proposed by  Ronneberger et  al. which consists  of  a  contracting  path  to  analyze  the  whole  image  and  a  symmetric expanding  path  to  recovery  the  original  resolution. The  U-Net  structure  has  been widely used in the field of medical image segmentationand has achieved competitive performance. Segmentation can be viewed to be a problem that balances global features in a local context
so we use transposed convolutions to better upsample learned feature maps along the expanding pathway. Severalstudies[14,  15]have demonstratedthat the  3D  versionsof  U-Net architecture using 3D volumes as input can produce better results than entirely 2D architecture. Although 3D U-Nethas good performance, it has more parameters than2D version, and the computational complexity of 3D  model is much higher than that of  2D  model.


---Unet pic

We use 3D convolutional network with a cascade and multi-stage framework to alleviate class imbalance which is the main di culty in brain tumor segmentation. The enhancing tumor makes _ %, TC _ % and _ and _ .We used the three stages for segmenting whole, core and enhancing tumor region into one cascade structure. In this way, the later stage network can focus on learning difficult features per class. We can also fit the model with reasonable GPU space and alter pre-processing for different tumor regions. We also use the insight of distribution split to change the architecture of of secondary networks. 


We found that significantly fewer resources for understanding image segmentation with CNNs are available, so to that end we made a video explaning incoming researchers about image segmentation in the U-Net: [An in-depth look at image segmentation in modern deep learning architectures through the UNet]
(https://www.youtube.com/watch?v=NzY5IJodjek)




