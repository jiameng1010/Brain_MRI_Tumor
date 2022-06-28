from Working_Environment.environment_variables import *
from BraTS_Data.BraTS_Data import BraTS_Data
from pipeline.pipeline import pipeline
from medpy.io import load
import subprocess
import nibabel as nib
import numpy as np
import os
if __name__ == "__main__":
    outputdir = 'DISP'


    for i in range(0, 369):
        test_data = BraTS_Data(BraTS_dataset_dir, i+1)
        #SamsegTumor_BraTS(test_data)
        if os.path.isfile(test_data.data_path + '/SamsegTumor_noskull_output/seg.mgz'):
            pp = pipeline(test_data.data_path + '/pipeline_folder', test_data.data_path + '/pipeline_folder/output',
                          1500,
                          is_synthesised=False, resume=True, disp_scale=0.0,
                          is_svf=False, use_BraTS_label=True)
            pp.project_samseg(src=test_data.data_path + '/SamsegTumor_noskull_output/seg.mgz',
                           dist=pp.output_dir + '/baseline.nii.gz',
                           direct_copy=True)
            pp.plot_brats_result('baseline.nii.gz',
                                 'samseg_TUMOR_DISP0.7.nii.gz',
                                 'samseg_TUMOR_DISP0.73.nii.gz',
                                 'plots/'+str(i)+'.png', save_volume=False, overlay=True)

