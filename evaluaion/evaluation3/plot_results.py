from Working_Environment.environment_variables import *
from BraTS_Data.BraTS_Data import BraTS_Data
from pipeline.pipeline import pipeline
from medpy.io import load
import subprocess
import nibabel as nib
import numpy as np
from eval_symmetry import Eval_symmetry
import os
if __name__ == "__main__":
    outputdir = 'DISP'

    for ii in range(10):
        dis_scale = 0.1 * ii * 0.6088199579835727#1.2063787751955717
        subprocess.run(['mkdir', outputdir+'/'+str(ii)])
        n = 0
        sum1 = 0
        sum2 = 0
        sum_increase = 0
        for i in range(0, 369):
            test_data = BraTS_Data(BraTS_dataset_dir, i+1)
            #SamsegTumor_BraTS(test_data)
            if os.path.isfile(test_data.data_path + '/SamsegTumor_noskull_output/seg.mgz'):
                '''subprocess.run(['mkdir', test_data.data_path + '/pipeline_folder'])
                subprocess.run(['cp', test_data.get_filename(3), test_data.data_path + '/pipeline_folder/image_0000.nii.gz'])
                subprocess.run(['cp', test_data.get_filename(0), test_data.data_path + '/pipeline_folder/image_0001.nii.gz'])
                subprocess.run(['cp', test_data.get_filename(1), test_data.data_path + '/pipeline_folder/image_0002.nii.gz'])
                subprocess.run(['cp', test_data.get_filename(2), test_data.data_path + '/pipeline_folder/image_0003.nii.gz'])
                subprocess.run(['cp', test_data.get_filename(4), test_data.data_path + '/pipeline_folder/image_seg.nii.gz'])'''
                n+=1
                pp = pipeline(test_data.data_path + '/pipeline_folder', test_data.data_path + '/pipeline_folder/output',
                              1500,
                              is_synthesised=False, resume=True, disp_scale=dis_scale,
                              is_svf=False, use_BraTS_label=True)
                pp.plot_result()
        print(n)

