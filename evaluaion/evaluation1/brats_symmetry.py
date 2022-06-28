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
    outputdir = 'subSVF_seg7'

    for ii in range(10):
        dis_scale = 0.1 * ii# * 0.6088199579835727#1.2063787751955717
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
                              is_svf=True, use_BraTS_label=True)
                from util_package.util import interpolate
                '''seg1 = nib.load(test_data.data_path + '/SamsegTumor_noskull_output/seg.mgz')
                tumor_seg = nib.load(test_data.data_path + '/SamsegTumor_noskull_output/posteriors/Tumor.mgz').get_fdata()
                output_image = seg1.get_fdata()
                output_image[np.where(tumor_seg==1)] = 99
                unwapped_seg = nib.Nifti1Image(output_image, pp.get_input_affine())
                nib.save(unwapped_seg, test_data.data_path + '/SamsegTumor_noskull_output/seg_with_tumor.mgz')'''

                displacement = pp.project_samseg(test_data.data_path + '/pipeline_folder/samseg7/seg.mgz',
                                                 test_data.data_path + '/pipeline_folder/samseg7/seg_warp.mgz',
                                                 is_inv=False, is_DISP=True)

                seg1 = nib.load(test_data.data_path + '/pipeline_folder/samseg7/seg.mgz')
                seg2 = nib.load(test_data.data_path + '/pipeline_folder/samseg7/seg_warp.mgz')
                eval1 = Eval_symmetry(seg1)
                eval2 = Eval_symmetry(seg2)
                print(eval1.get_mean_dice())
                sum1 += eval1.get_mean_dice()
                print(eval2.get_mean_dice())
                sum2 += eval2.get_mean_dice()

                increase = (eval2.get_mean_dice() - eval1.get_mean_dice()) / eval1.get_mean_dice()
                sum_increase += increase
                print('--------------------')

                text_f = open(outputdir+'/'+str(ii)+'/'+str(i+1)+'.txt', 'w')
                text_f.write(str(eval1.get_mean_dice()) + '\n')
                text_f.write(str(eval2.get_mean_dice()) + '\n')
                text_f.write(str(eval1.get_mean_dice(weighted=False)) + '\n')
                text_f.write(str(eval2.get_mean_dice(weighted=False)) + '\n')
                text_f.close()
        print(n)
        print(sum1/n)
        print(sum2/n)
        sum_increase += increase

