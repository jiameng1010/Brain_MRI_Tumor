from Working_Environment.environment_variables import *
from BraTS_Data.BraTS_Data import BraTS_Data
from pipeline.pipeline import pipeline
from medpy.io import load
import subprocess
import nibabel as nib
import numpy as np
from pipeline.plot_result import Plot_result
from eval_symmetry import Eval_symmetry
import os


class pipeline_plot_synthesized(pipeline):
    def plot_synthesized(self, pred_file1, pred_file2, output_file):
        metric1 = self.run_eval(pred_file1)
        metric2 = self.run_eval(pred_file2)
        metric1_nocortex = self.run_eval(pred_file1, with_cortex=False)
        metric2_nocortex = self.run_eval(pred_file2, with_cortex=False)
        mri = nib.load(self.input_dir + '/sythesised_t1_0001.nii.gz').get_fdata()
        mri_warp = nib.load(self.output_dir + '/wapped_t1.nii.gz').get_fdata()
        pred1 = nib.load(self.output_dir + '/' + pred_file1).get_fdata()
        pred2 = nib.load(self.output_dir + '/' + pred_file2).get_fdata()
        gt = nib.load(self.input_dir + '/transformed_labels_manual_aseg.nii.gz').get_fdata()
        affine = nib.load(self.input_dir + '/transformed_labels_manual_aseg.nii.gz').affine
        gt[np.where(gt == 45)] = 47.0
        gt[np.where(gt == 6)] = 8.0
        pred1[np.where(pred1 == 41)] = 0.0
        pred1[np.where(pred1 == 2)] = 0.0
        pred1[np.where(pred1 ==99)] = 0.0
        pred2[np.where(pred2 == 41)] = 0.0
        pred2[np.where(pred2 == 2)] = 0.0
        pred2[np.where(pred2 == 99)] = 0.0
        plot = Plot_result(mri, mri_warp, pred1, pred2, gt, affine)
        plot.produce_img(output_file, metric1, metric2, metric1_nocortex, metric2_nocortex)
        return 0


if __name__ == "__main__":
    data_dir =