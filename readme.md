In Progress: Some files are old and update will be coming soon.

## Abstract

A models's lower bound performance is a vital metric in determining trust in practical clinical settings that is often overlooked in medical research and competitions. In medical datasets the data may have different distributions due to it being collected from multiple sources. We handle the challenge of a distribution split in a dataset by a series of models trained on different loss functions, each handling an aspect of the imbalance, to create an ensemble that provides the best worst-case performance on the BraTs 2018 dataset. For the task of increasing the lower bound on performance we propose a simple modification to the well known Dice loss called Power Loss to penalize samples far away from the average in a non-linear fasion. This formulation handles the distribution imbalance by preventing the model from over fixating on the major distribution and bringing comparable performace to the minor distributions. We combine our loss with variants of the Tversky loss to create a ensemble that achives homogeneous performance across the validation set.

This repository contains some files from MICCA BraTS segmentation challenge: https://www.med.upenn.edu/sbia/brats2018/data.html

## Table of Contents
1. [Dataset](#dataset)
2. [Data Pre-Processing ](#Data-Pre-Processing )
3. [Convolutional Neural Networks](#Convolutional-Neural-Networks)
    * [Model Architecture](#model-architecture)
    * [Training the Model](#training-the-model)  
    * [Patch Selection](#patch-selection)
    * [Results](#results)
4. [Ensembling and Loss Function](#Ensembling-and-Loss-Function)


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

## Data Pre-Processing 
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
so we use transposed convolutions to better upsample learned feature maps along the expanding pathway. Several studies have demonstrated that the  3D  versionsof  U-Net architecture using 3D volumes as input can produce better results than entirely 2D architecture. Although 3D U-Net has good performance, it has significantly more parameters than the 2D version, and therfore higher computational complexity. 




<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/%20%20UnetWholeAnimation.png?raw=true" width="900" height="500" >
</div>


We use 3D convolutional network with a cascade and multi-stage framework to alleviate class imbalance which is the main difficulty in brain tumor segmentation. The enhancing tumor makes _ %, TC _ % and _ and _ .We used the three stages for segmenting whole, core and enhancing tumor region into one cascade structure. In this way, the later stage network can focus on learning difficult features per class. We can also fit the model with reasonable GPU space and alter pre-processing for different tumor regions. We also use the insight of distribution split to change the architecture of of secondary networks. Due to the large differences in distribution and standard deviations between the modalities, we pass each modality from the region proposal into an additional InputBlock before concatenating them to be processed by the network jointly. The InputBlock is composed of three convolution layers which takes each modality from 1 channel to 4 channels and after concatenation a 16 channel input is passed into the secondary unet of the same structure as before.

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/InputBlockAni.png?raw=true" width="320" height="200" >
</div>

Our first network segments the whole tumor and provides a region proposal that is fed into the later cascades. This reduced the memory footprint of the cascades and shows only the relevant regions for quicker learning. Additionally we found using ELU activation function superior to ReLU. For data augmentations we do rotations, random flips and add gaussian noise along with a guassian smoothing filter. We found batch normalization to not work well with the small batch sizes we trained on so we used Instance Norm with much sucess. We use [DropBlock](https://arxiv.org/abs/1810.12890) regularization instead of Dropout.We also use residual connections in each block to facilitate learning.

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/residual.png?raw=true" width="250" height="250" >
</div>


We found that significantly fewer resources for understanding image segmentation with CNNs are available, so to that end we made a video explaning incoming researchers about image segmentation in the U-Net: [An in-depth look at image segmentation in modern deep learning architectures through the UNet](https://www.youtube.com/watch?v=NzY5IJodjek)


## Ensembling and Loss Function

As the data has two seperate distributions, training a model with a single loss function leads the model to favor one data distribution over the other. This causes the validation scores to be split among very high results and a few low ones. While this is fine in a competitions where our objective is to maximize the average score, this erodes the trust of our algorithms in practical settings where consistecy is more important that a few points on the average. To this end, we create a diverse set of models using different variants of the dice loss to establish a lower bound on validation performance so our models can be trusted in practical settings.

### Dice Loss
We use a combination of the cross entropy loss and variations of dice loss for training. The dice loss directly optimizes the IoU metric and is know for its superior performance because of how it optimizes the area in segmentation tasks. The dice loss is given by 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{L}_{d&space;i&space;c&space;e}=1-\mathbf{Dice}_{coef}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{L}_{d&space;i&space;c&space;e}=1-\mathbf{Dice}_{coef}" title="\mathbf{L}_{d i c e}=1-\mathbf{Dice}_{coef}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{L}_{d&space;i&space;c&space;e}=1-\frac{2&space;*&space;\sum&space;p_{t&space;r&space;u&space;e}&space;*&space;p_{p&space;r&space;e&space;d}}{\sum&space;p_{t&space;r&space;u&space;e}^{2}&plus;\sum&space;p_{p&space;r&space;e&space;d}^{2}&plus;\epsilon}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{L}_{d&space;i&space;c&space;e}=1-\frac{2&space;*&space;\sum&space;p_{t&space;r&space;u&space;e}&space;*&space;p_{p&space;r&space;e&space;d}}{\sum&space;p_{t&space;r&space;u&space;e}^{2}&plus;\sum&space;p_{p&space;r&space;e&space;d}^{2}&plus;\epsilon}" title="\mathbf{L}_{d i c e}=1-\frac{2 * \sum p_{t r u e} * p_{p r e d}}{\sum p_{t r u e}^{2}+\sum p_{p r e d}^{2}+\epsilon}" /></a>

While the Cross Entropy Loss is a per pixel loss, the Dice Loss takes a proportion of predicted and true areas which causes it to behave differently in a few cases. While the Cross Entropy and Dice Loss theoretically try to optimize the same objective, the dice loss gives more importance to samples which are less visually consistent by taking the ratio of areas. So, while the Cross Entropy would penalise a small and large target region with only some X pixels incorrect in the same manner, the dice will be more lenient for the larger target as most of the region has been predicted.

### Tversky Loss
An equivalent representation is using True/False Positives/Negatives:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{L}_{d&space;i&space;c&space;e}&space;=1-&space;\frac{2TP}{2TP&plus;FN&plus;FP}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{L}_{d&space;i&space;c&space;e}&space;=1-&space;\frac{2TP}{2TP&plus;FN&plus;FP}" title="\mathbf{L}_{d i c e} =1- \frac{2TP}{2TP+FN+FP}" /></a>

The above expression allows us to weight the false positives or negatives by adding a multiplicative factor to them to account for over or under segmenting. This is the motivation of the [Tversky Loss:](https://arxiv.org/abs/1706.05721)

<a href="https://www.codecogs.com/eqnedit.php?latex=T(\alpha,&space;\beta)=1&space;-&space;\frac{\sum_{i=1}^{N}&space;p_{0&space;i}&space;g_{0&space;i}}{\sum_{i=1}^{N}&space;p_{0&space;i}&space;g_{0&space;i}&plus;\alpha&space;\sum_{i=1}^{N}&space;p_{0&space;i}&space;g_{1&space;i}&plus;\beta&space;\sum_{i=1}^{N}&space;p_{1&space;i}&space;g_{0&space;i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(\alpha,&space;\beta)=1&space;-&space;\frac{\sum_{i=1}^{N}&space;p_{0&space;i}&space;g_{0&space;i}}{\sum_{i=1}^{N}&space;p_{0&space;i}&space;g_{0&space;i}&plus;\alpha&space;\sum_{i=1}^{N}&space;p_{0&space;i}&space;g_{1&space;i}&plus;\beta&space;\sum_{i=1}^{N}&space;p_{1&space;i}&space;g_{0&space;i}}" title="T(\alpha, \beta)=1 - \frac{\sum_{i=1}^{N} p_{0 i} g_{0 i}}{\sum_{i=1}^{N} p_{0 i} g_{0 i}+\alpha \sum_{i=1}^{N} p_{0 i} g_{1 i}+\beta \sum_{i=1}^{N} p_{1 i} g_{0 i}}" /></a>

Unlike random forest and other high variance classifiers, neural networks are relatively low variance. I.e models trained again and again dont achive much variance in their outputs. Thus we must artifically add factors that differenciate one model from the next so we have a diverse set of models to take advantage of. The above forumlation of the dice allows us to alter the FN and FP weights as hyperparameters to create an array of models that each balance over and under segmentating differently.  

### Power Loss

The dice loss is a very linear loss function. If we raise our dice coefficent to a power we can consider what the gradient would look like:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{L}_{d&space;i&space;c&space;e}=1-\mathbf{Dice}_{coef}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{L}_{d&space;i&space;c&space;e}=1-\mathbf{Dice}_{coef}^n" title="\mathbf{L}_{d i c e}=1-\mathbf{Dice}_{coef}^n" /></a>

Taking the gradient with respect to any paramater yields:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d\mathbf{L}_{d&space;i&space;c&space;e}}{dw_i}=-n\mathbf{Dice}^{n-1}_{coef}&space;\frac{d\mathbf{Dice}_{coef}}{dw_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d\mathbf{L}_{d&space;i&space;c&space;e}}{dw_i}=-n\mathbf{Dice}^{n-1}_{coef}&space;\frac{d\mathbf{Dice}_{coef}}{dw_i}" title="\frac{d\mathbf{L}_{d i c e}}{dw_i}=-n\mathbf{Dice}^{n-1}_{coef} \frac{d\mathbf{Dice}_{coef}}{dw_i}" /></a>

We see that the gradient term is multiplied by a factor proportional to the loss at that point. So hard samples yielding higher losses will get weighted more in this formulation automatically allowing our model to focus on the minority distribution in the dataset. This a similar forumulation to the M.S.E loss which penalizes distances in a non-linear fasion. 

### Ensemble

Using 8 models trained on each of the above, we create a ensemble that is able to overcome the data distribution split problem and achives consistent results throughout samples. 

* 2 models on Dice Loss

* 2 Models on Tversky loss focusing on False Negatives

* 2 models on Power Loss with n=2

## Results


 Method         | Dice ET   |  Dice WT  | Dice TC | 
--------------- |:---------:|:---------:|:--------:|
 *Ours*         | 0.74      |   0.89    |   0.81   |  
 
 
 ### Worst Case Performance and Variance of models
 
 We calculate the worst case scores and standard deviations across Dice, Precision and Recall Scores in the validation set. We find our ensemble to do the best in all cases but one and models trained on power loss to do better in handling the distribution split that the rest.
 
### Worst Case Scores
 
 
   Loss       |   Worst Case Dice |  Worst Case Precision | Worst Case Recall
--------------- |:---------:| :---------:|:--------:|
 Tversky - 1    | 0.649     |  0.581      | 0.532|
 Tversky - 2    | 0.682     |  0.605      | 0.543|
 Dice - 1       | 0.676     | 0.643       | 0.543|
 Dice - 2       | 0.642     |  0.674      | 0.494|
 Power Loss - 1 | 0.736     |  0.583      | **0.721**|
 Power Loss - 2 | 0.772     | 0.642       | 0.674|
 Ensemble       | **0.795** |  **0.739**  | 0.719|
 
 
 #### Standard Deviations

  Loss         |    Dice   |  Precision  | Recall | 
--------------- |:---------:|:---------:|:--------:|
 Tversky - 1    | 0.0749     |   0.0928    |   0.0974   |  
 Tversky - 2    | 0.0670     |   0.0887    |   0.0918   |  
 Dice - 1       | 0.0573      |   0.0700    |   0.0867   |  
 Dice - 2       | 0.0602     |   0.0722    |   0.0949   |  
 Power Loss - 1 | 0.0518      |   0.0869   |   0.0658   |  
 Power Loss - 2 | 0.0479      |   0.0741    |   0.0735   |  
 Ensemble       | **0.0391**    | **0.0589**   |   **0.0639**   |  
 

 
## Validation Samples
The first four images are the the four modalities(flair, t1, t1ce and t2 respectively, the next image is the network prediction and the last image is the ground truth.

### Whole Tumor

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/whole1.png?raw=true" width="900" height="180" >
</div>

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/whole3.png?raw=true" width="900" height="180" >
</div>

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/whole4.png?raw=true" width="900" height="180" >
</div>



### Core Tumor

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/core3.png?raw=true" width="900" height="180" >
</div>

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/core1.png?raw=true" width="900" height="180" >
</div>

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/core4.png?raw=true" width="900" height="180" >
</div>



### Enhancing Tumor

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/enhancing.png?raw=true" width="900" height="180" >
</div>

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/enhanc1.png?raw=true" width="900" height="180" >
</div>

<div id="container">
    <img src="https://github.com/bluesky314/BraTs-Segmentation-Challenge/blob/master/images/enchanc2.png?raw=true" width="900" height="180" >
</div>

