In Progress: Some files are old and update will be coming soon.

This repository contains some files from MICCA BraTS segmentation challenge: https://www.med.upenn.edu/sbia/brats2018/data.html


## Dataset
BraTS 2018 utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. All MRI data was provided by the [2018 MICCAI BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/tasks.html), which consists of 210 high-grade glioma cases and 75 low-grade cases. Each patient has four different MRI [modalities](#modalities), each of which is comprised of 155 brain slices, for a total of 620 images per patient. Professional segmentation is provided as ground truth labels for each case. 

The three tumor regions consist of the whole tumor (yellow), the tumor core (red), the enhancing tumor structures(blue). Due to this highly diverse appearance and shape, segmentation of brain tumors in multimodal MRI scans is one of the most challenging tasks in medical image analysis. This intrinsic diversity of gliomas is also portrayed in their imaging features (appearance and shape), and their sub-regions are described by varying intensity profiles spread across multimodal MRI scans, reflecting varying tumor biological properties. 


<div id="container">
    <img src="https://www.med.upenn.edu/sbia/assets/user-content/BRATS_tasks.png?raw=true" width="550" height="250" >
</div>

Different Tumor Classes and Modalities

Shown above are image patches that are annotated witht the different types of classes in the different modalities (top left) and the final labels for the whole dataset (right). The image patches show from left to right: the whole tumor (yellow) visible in T2-FLAIR (Fig.A), the tumor core (red) visible in T2 (Fig.B), the enhancing tumor structures (light blue) visible in T1Gd, surrounding the necrotic components of the core (green) (Fig. C). The segmentations are combined to generate the final labels of the tumor sub-regions.


## Data pre-processing 
One of the challenges in working with MRI data is dealing with the artifacts produced either by inhomogeneity in the magnetic field or small movements made by the patient during scan time. Oftentimes a bias will be present across the resulting scans, which negatively affect the segmentation results.  We notice that out of 285 total patients, 210 are not much affected by bias correction however 75 are greatly affected indicating a split in distribution in the dataset. Bias correcting raises results on these substantially however it must be notes to not bias correct flair. Many papers recommed different stratgies for bias correction with no consistency. Bias correcting flair will shift its distribution to negatively impact the tumur enhancing and core. We learnt this the hard way. 

<div id="container">
    <img src="https://github.com/naldeborgh7575/brain_segmentation/blob/master/images/n4_correction.png?raw=true" width="300" height="150" >
</div>

We employ SITK's bias correction on all T1,T2 and T1CE images.  which removed the intensity gradient on each scan. Additional image pre-processing requires standardizing the pixel intensities, since MRI intensities are expressed in arbitrary units and may differ significantly between machines used and scan times. We remove as much of background pixels as possible and crop each image to (128,128,172). We normalise by removing the top 2% pixel value and by median as there is alot of high frequency noise in the data. We also rescale each image to (0,1). We found this greatly speeds up learning.

<div id="container">
    <img src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/brain_grids.png?raw=true" width="900" height="500" >
</div>

<div id="container">
    <img src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/segment.png?raw=true" width="900" height="500" >
</div>
