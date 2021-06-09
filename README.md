# Brain_MRI_Tuomr
This project explores the possibility of remove (by warping) the pathology area of a tumor-presenting brain MR image, so that an automatic sub-cortical segmentation method designed form healthy brain can handle it.

## Working Environment
**0 - Prerequisites**
This setup instruction assume you are working on an ubuntu 18.04 or 20.04 system. 
The simplest way to setup all dependencies for this project is using anaconda. If you haven't install conda in your system, please first do so by following the instruction in https://docs.anaconda.com/anaconda/install/linux/

**1 - Conda Environmrnt**
If you have already installed conda, create a new environment for this project by
```
conda env create -f Working_Environmrnt/volume_segmentation_environment.yml
```
`volume_segmentation_environment.yml` is a list of required packages that could be installed by conda. I will keep update this file through this project.
After the environment is created, activate it by `conda activate volume_segmentation`.

**2 - Pymesh**
