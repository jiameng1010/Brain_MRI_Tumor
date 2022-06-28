#!/bin/bash
export FREESURFER_HOME=/home/mjia/Working_Tools/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/00019/
recon-all -subjid sythesised_t1 -i /home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/00019/sythesised_t1_0001.nii.gz -autorecon1 -gcareg