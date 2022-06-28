
from Working_Environment.environment_variables import *
from BraTS_Data.BraTS_Data import BraTS_Data
from pipeline.pipeline import pipeline
from medpy.io import load
import subprocess
import nibabel as nib
import numpy as np
import os

test_data_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/'


for i in range(40):
    input_folder = test_data_dir + str(i).zfill(5)
    pp = pipeline(input_folder, input_folder +'/output', 1500,
                  is_synthesised=True, resume=True, disp_scale=0.0, is_svf=False)

    '''source = i%20
    subprocess.run(['cp', '-r',
                    '/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/synthesised7_old/'+str(source).zfill(5)+'/sythesised_t1',
                    input_folder])

    pp.tumor_segmentation()
    pp.produce_point_cloud()
    pp.pred_svf()
    pp.pred_disp()'''