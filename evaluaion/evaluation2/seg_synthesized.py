
from Working_Environment.environment_variables import *
from BraTS_Data.BraTS_Data import BraTS_Data
from pipeline.pipeline import pipeline
from medpy.io import load
import subprocess
import nibabel as nib
import numpy as np
import os, sys
sys.path.append(SAMSEG_DIR)
from freesurfer.samseg import Samseg, ProbabilisticAtlas, SamsegTumor

test_data_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/'

#SamsegTumor.makeTumorAtlas(SAMSEG_ATLAS_DIR1, SAMSEG_ATLAS_TUMOR_DIR, skull=False)
for i in range(0, 40):

    input_folder = test_data_dir + str(i).zfill(5)
    pp = pipeline(input_folder, input_folder +'/output', 1500,
                  is_synthesised=True, resume=True, disp_scale=0.45, is_svf=False)###################

    '''samseg_dir = pp.input_dir + '/samseg1'
    if os.path.isdir(samseg_dir):
        subprocess.run(['rm', '-r', samseg_dir])
    subprocess.run(['mkdir', samseg_dir])
    ss = SamsegTumor.SamsegTumor(imageFileNames=[pp.t1],
                                 atlasDir=SAMSEG_ATLAS_TUMOR_DIR,
                                 savePath=samseg_dir,
                                 modeNames=['t1'],
                                 useMeanConstr=False, useShapeModel=False, useMRF=False,
                                 savePosteriors=True)
    ss.segment()
    tag = 'Samseg'''''

    samseg_dir = pp.input_dir + '/samseg3'#######################
    if pp.is_svf:
        tag = 'SVF_method3'###########################
    else:
        tag = 'DISP_method3'#############################
    pp.samseg_tumor_def(pp.t1, samseg_dir, use_GT_disp=False, use_deformed_atlas=True)



    pp.project_samseg(src=samseg_dir + '/seg.mgz',
                        dist=pp.output_dir + '/samseg_tumor_' + tag + str(pp.tau)[:4] + '.nii.gz',
                        direct_copy=True)

    #seg1 = nib.load(samseg_dir + '/seg.mgz')
    #tumor_seg = nib.load(samseg_dir + '/posteriors/Tumor.mgz').get_fdata()
    #output_image = seg1.get_fdata()
    #output_image[np.where(tumor_seg == 1)] = 99
    #unwapped_seg = nib.Nifti1Image(output_image, pp.get_input_affine())
    #nib.save(unwapped_seg, samseg_dir + '/seg_with_tumor.mgz')
    pp.project_samseg(src=samseg_dir + '/segT.mgz',
                        dist=pp.output_dir + '/samseg_TUMOR_' + tag + str(pp.tau)[:4] + '.nii.gz',
                        direct_copy=True)