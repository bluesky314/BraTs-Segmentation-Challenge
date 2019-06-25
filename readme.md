This repository contains some files from MICCA BraTS segmentation challenge: https://www.med.upenn.edu/sbia/brats2018/data.html

Files are quite old and update will be coming soon.

## Dataset
BraTS 2018 utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. All MRI data was provided by the [2018 MICCAI BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/tasks.html), which consists of 210 high-grade glioma cases and 75 low-grade cases. Each patient has four different MRI [modalities](#modalities), each of which is comprised of 155 brain slices, for a total of 620 images per patient. Professional segmentation is provided as ground truth labels for each case. 

Due to this highly diverse appearance and shape, segmentation of brain tumors in multimodal MRI scans is one of the most challenging tasks in medical image analysis. This intrinsic diversity of gliomas is also portrayed in their imaging features (appearance and shape), and their sub-regions are described by varying intensity profiles spread across multimodal MRI scans, reflecting varying tumor biological properties. 


<div id="container">
    <img src="https://www.med.upenn.edu/sbia/assets/user-content/BRATS_tasks.png?raw=true" width="550" height="250" >
    <center>Incremental Learning rules</center>
</div>


