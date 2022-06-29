# Brain_MRI_Tumor
This project explores the possibility of remove (by warping) the pathology area of a tumor-presenting brain MR image, so that an automatic sub-cortical segmentation method designed form healthy brain can handle it.

## Working Environment
**0 - Prerequisites**
This setup instruction assume you are working on an ubuntu 18.04 or 20.04 system. 
The simplest way to setup all dependencies for this project is using anaconda. If you haven't install conda in your system, please first do so by following the instruction in https://docs.anaconda.com/anaconda/install/linux/

**1 - Conda Environmrnt**
If you have already installed conda, create a new environment for this project by
```
conda env create -f Working_Environment/volume_segmentation_environment.yml
```
`volume_segmentation_environment.yml` is a list of required packages that could be installed by conda. I will keep update this file through this project.
After the environment is created, activate it by `conda activate volume_segmentation`.

**2 - Pymesh**
Pymesh is a powerful python tool for processing mesh data. To make it working properly, one have to compile it and its dependencies in local. The download like and install instruction could be found here:
https://pymesh.readthedocs.io/en/latest/installation.html
This project use Pymesh to do many mesh operations. One of the most important is tetrahedralizing surface meshes. 

**3 - FreeSurfer**
FreeSufer is an essential software tool for processing medical Images. You can find its download and install instruction here:
https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
This project use FreeSurfer to make affine registration (images from different datasets to a common brain space).

**4 - TumorSim**
TumorSim is a cross-platform simulation software that generates pathological ground truth from a healthy ground truth. You can find its download and install instruction here:
https://www.nitrc.org/projects/tumorsim/
We relay on this software to generate training data.
Edit ```TumorSim_executable``` in Working_Environment/environment_variables.py to point to the binary executable of TumorSim.

**5 - SamsegTumor**
The original SamsegTumor.py in freesurfer/python/packages/freesurfer/samseg should be replaced by our modified version at Modified_SamsegTumor/SamsegTumor.py.

**6 - PCNN displacement network**
The code for training the PCNN displacement network is in another repo: https://github.com/jiameng1010/pcnn_brain_tumor.

## Datasets
**1 - BraTS**
BraTS 2020 dataset could be download from:
https://www.med.upenn.edu/cbica/brats2020/data.html
Once extracted, edit ```BraTS_dataset_dir``` in Working_Environment/environment_variables.py to point to where BraST is stored.

**2 - Mindboggle**
Mindboggle-101 dataset could be download from:
https://mindboggle.info/data
Once extracted, edit ```Mindboggle_dataset_dir``` in Working_Environment/environment_variables.py to point to where Mindboggle-101 is stored.

**3 - TumorSim Input data**
Some synthetic brain MR data is released together with TumorSim software package. You should download #1, #2, #3, #4, and #5 of the data from:
https://www.nitrc.org/frs/?group_id=546
As the same, edit ```BrainSim_inputdata_dir``` in Working_Environment/environment_variables.py to point to where these files are stored.
