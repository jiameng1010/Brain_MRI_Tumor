from medpy.io import load, save
import nibabel as nib
import numpy as np
import copy

def segmentation_geometric_transformation(source, affine_in, shape_out, displacement=None):
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from util_package.util import clap_voxels_out, affine
    output_image = np.zeros(shape=shape_out, dtype=source.dtype)
    index = np.where(output_image == 0)
    voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                         np.expand_dims(index[1], axis=1),
                                         np.expand_dims(index[2], axis=1)], axis=1)
    voxels_affine = affine(affine_in, voxels_interpolate)
    if not displacement is None:
        voxels_affine = voxels_affine + np.reshape(displacement, (-1, 3))
    shape = source.shape
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    z = np.linspace(0, shape[2] - 1, shape[2])
    fn = RegularGridInterpolator((x, y, z), source, method='nearest')
    voxels_in_displacement, index = clap_voxels_out(voxels_affine, index, shape)
    output = fn(voxels_in_displacement)
    output_image[index] = output
    return output_image

left_structures = [3, 77, 31, 4, 11, 12, 26, 13, 28, 18, 30, 10, 17]
right_structures = [42, 43, 50, 63, 51, 58, 52, 60, 54, 62, 49, 53, 63]
left_structures = [3, 4, 5, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
right_structures = [42, 43, 44, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]
#left_structures = [3]
#right_structures = [42]

class Eval():
    def __init__(self, pred, gt, affine_matrix, tumor_side=None, with_cortex=True, tumor_at_left=None):
        self.pred = pred
        gt[np.where(gt>=2000)] = 42.0
        gt[np.where(gt>=1000)] = 3.0
        gt[np.where(gt == 45)] = 47.0
        gt[np.where(gt == 6)] = 8.0
        self.GT = gt
        self.rotation = affine_matrix[:3, :3]
        self.with_cortex = with_cortex
        #self.eliminate_small()

        if tumor_at_left==None:
            tumor_at_left = self.get_tumor_center()

        if tumor_side==None:
            #cerebral labels + Brain-Stem + CSF
            self.labels = left_structures + right_structures# + [7, 8, 46, 47] + [16, 24]
        elif tumor_side:
            if not tumor_at_left:
                self.labels = copy.copy(right_structures)
            else:
                self.labels = copy.copy(left_structures)
        else:
            if tumor_at_left:
                self.labels = copy.copy(right_structures)
            else:
                self.labels = copy.copy(left_structures)

        if not self.with_cortex:
            if 3 in self.labels:
                self.labels.remove(3)
            if 42 in self.labels:
                self.labels.remove(42)

        self.calculate_Dice()
        self.error_distributation()

    def eliminate_small(self):
        from scipy.ndimage.morphology import binary_erosion, binary_opening
        import copy
        diff = np.zeros_like(self.GT)
        diff[np.where(self.GT != self.pred)] = 1
        diff_erosion = copy.copy(diff)
        diff_erosion = 1 * binary_opening(diff_erosion)
        mask = diff - diff_erosion
        self.pred[np.where(mask == 1)] = -2
        self.GT[np.where(mask == 1)] = -2

    def error_distributation(self):
        import scipy.ndimage
        self.diff = np.zeros_like(self.GT)
        self.diff[np.where(self.GT != self.pred)] = 1
        #print(np.sum(self.diff))
        #seed_out, num_features = scipy.ndimage.measurements.label(self.diff)
        #print(num_features)
        #print(np.sum(self.diff) / num_features)
        return 0

    def get_error(self):
        return np.sum(self.diff)

    def get_tumor_center(self):
        tumor_voxel_xyz = np.where(self.GT==-1)
        stem_voxel_xyz = np.where(self.GT == 16)
        self.tumor_center = np.mean(tumor_voxel_xyz, axis=1)
        tumor_certer_world = self.rotation.dot(np.mean(tumor_voxel_xyz, axis=1))
        stem_center_world = self.rotation.dot(np.mean(stem_voxel_xyz, axis=1))
        return tumor_certer_world[0] <= stem_center_world[0]

    def weighted_by_distance(self, label):
        voxel_xyz = np.where(self.GT == label)
        difference = np.asarray(np.asarray(voxel_xyz).T - self.tumor_center)
        distance = np.sqrt(np.sum(difference * difference, axis=1))
        return len(voxel_xyz[0]) / np.mean(distance)

    def calculate_Dice(self):
        self.dice = []
        self.weights1 = []
        self.weights2 = []
        for label in self.labels:
            self.dice.append(self.calculate_Dice_one(label))
            self.weights1.append(len(np.where(self.GT == label)[0])+1)
            #self.weights1.append(1)
            #self.weights2.append(self.weighted_by_distance(label)+1)

        self.dice = np.asarray(self.dice)
        weights1 = np.asarray(self.weights1)
        self.weights1 = weights1 / np.sum(weights1)
        weights2 = np.asarray(self.weights2)
        self.weights2 = weights2 / np.sum(weights2)
        return 0

    def calculate_Dice_one(self, label):
        from scipy.spatial.distance import dice
        pred = np.zeros_like(self.pred)
        pred[np.where(self.pred == label)] = 1
        GT = np.zeros_like(self.GT)
        GT[np.where(self.GT == label)] = 1
        intersection = np.logical_and(pred, GT)
        union = np.logical_or(pred, GT)
        if pred.sum() + GT.sum() == 0:
            return 1.0
        else:
            return 2 * intersection.sum() / (pred.sum() + GT.sum())

    def get_dice(self):
        return self.dice

    def get_weighted_mean_dice1(self):
        #print(np.sum(self.weights1 * self.dice))
        return np.sum(self.weights1 * self.dice)

    def get_weighted_mean_dice2(self):
        return np.sum(self.weights2 * self.dice)

import subprocess
from Working_Environment.environment_variables import *
def segmentation_samseg(input_file, output_folder):
    command = ['run_samseg',
               '-o', output_folder,
               '-i', input_file]

    subprocess.run(['rm', 'FreeSurfer_script.sh'])
    with open('FreeSurfer_script.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('export FREESURFER_HOME=' + FREESURFER_HOME + '\n')
        f.write('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
        f.write(" ".join(command))

    subprocess.run(['chmod', '777', './FreeSurfer_script.sh'])
    subprocess.check_output(['./FreeSurfer_script.sh'])

if __name__ == '__main__':

    list_data = ['00000',
                 '00001',
                 '00002',
                 '00003',
                 '00004',
                 '00005',
                 '00006',
                 'my_training_000',
                 'my_training_001',
                 'my_training_002',
                 'my_training_003',
                 'my_training_seed_003',
                 'my_training_seed_004',
                 'my_training_seed_005',
                 'my_training_seed_006',
                 'my_training_seed_007']

    means = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for data in list_data:
        print(data)
        data_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised/' + data

        segmentation_samseg(data_dir+'/sythesised_t1_0001.nii.gz', data_dir + '/samseg1')
        segmentation_samseg(data_dir + '/output/wapped_t1.nii.gz', data_dir + '/samseg2')

        tumor_seg_file = nib.load(data_dir+'/tumor_segmentation/sythesised_t1.nii.gz')
        tumor_seg = tumor_seg_file.get_fdata()

        pred1 = nib.load(data_dir+'/output/fs_seg1.nii.gz')
        sf_pred1 = pred1.get_fdata()
        pred2 = nib.load(data_dir+'/output/fs_seg2.nii.gz')
        sf_pred2 = pred2.get_fdata()
        pred1 = nib.load(data_dir+'/output/atlas_seg1.nii.gz')
        atlas_pred1 = pred1.get_fdata()
        pred2 = nib.load(data_dir+'/output/atlas_seg2.nii.gz')
        atlas_pred2 = pred2.get_fdata()

        GT = nib.load(data_dir + '/transformed_labels_manual_aseg.nii.gz')
        gt = GT.get_fdata()
        gt[np.where(gt>=2000)] = 42
        gt[np.where(gt>=1000)] = 3
        '''from scipy.ndimage.morphology import binary_dilation
        tumor_mask = 1 * (gt == -1)
        tumor_mask = 1 * binary_dilation(tumor_mask)
        gt[np.where(tumor_mask == 1)] = -1'''

        sf_pred1[np.where(gt == -1)] = -1
        sf_pred2[np.where(gt == -1)] = -1
        atlas_pred1[np.where(gt == -1)] = -1
        atlas_pred2[np.where(gt == -1)] = -1

        '''sf_pred1[np.where(tumor_seg >= 2)] = -1
        sf_pred2[np.where(tumor_seg >= 2)] = -1
        atlas_pred1[np.where(tumor_seg >= 2)] = -1
        atlas_pred2[np.where(tumor_seg >= 2)] = -1'''

        eval1 = Eval(atlas_pred1, gt)
        #print(eval1.get_weighted_mean_dice1())
        #print(eval1.get_weighted_mean_dice2())
        #print('____________')
        eval2 = Eval(atlas_pred2, gt)
        #print(eval2.get_weighted_mean_dice1())
        #print(eval2.get_weighted_mean_dice2())
        #print('____________')
        print(eval2.get_weighted_mean_dice1() - eval1.get_weighted_mean_dice1())
        means[0] += eval2.get_weighted_mean_dice1() - eval1.get_weighted_mean_dice1()
        print(eval2.get_weighted_mean_dice2() - eval1.get_weighted_mean_dice2())
        means[1] += eval2.get_weighted_mean_dice2() - eval1.get_weighted_mean_dice2()
        print(eval1.get_error() - eval2.get_error())
        means[2] += eval1.get_error() - eval2.get_error()
        eval3 = Eval(sf_pred1, gt)
        #print(eval3.get_weighted_mean_dice1())
        #print(eval3.get_weighted_mean_dice2())
        #print('____________')
        eval4 = Eval(sf_pred2, gt)
        #print(eval4.get_weighted_mean_dice1())
        #print(eval4.get_weighted_mean_dice2())
        #print('____________')
        print(eval4.get_weighted_mean_dice1() - eval3.get_weighted_mean_dice1())
        means[3] += eval4.get_weighted_mean_dice1() - eval3.get_weighted_mean_dice1()
        print(eval4.get_weighted_mean_dice2() - eval3.get_weighted_mean_dice2())
        means[4] += eval4.get_weighted_mean_dice2() - eval3.get_weighted_mean_dice2()
        print(eval3.get_error() - eval4.get_error())
        means[5] += eval3.get_error() - eval4.get_error()
        print('____________')
        print(means)

        '''from scipy.ndimage.morphology import binary_erosion
        error_volume = np.zeros_like(gt)
        error_volume[np.where(eval1.get_error() == 1)] = 1
        error_volume = 1.0 * binary_erosion(error_volume)
        error_volume = 1.0 * binary_erosion(error_volume)
        nib.save(nib.Nifti1Image(error_volume, GT.affine), data_dir+'/error_vol1.nii.gz')
        error_volume = np.zeros_like(gt)
        error_volume[np.where(eval2.get_error() == 1)] = 1
        error_volume = 1.0 * binary_erosion(error_volume)
        error_volume = 1.0 * binary_erosion(error_volume)
        nib.save(nib.Nifti1Image(2*error_volume, GT.affine), data_dir+'/error_vol2.nii.gz')
        error_volume = np.zeros_like(gt)
        error_volume[np.where(eval3.get_error() == 1)] = 1
        error_volume = 1.0 * binary_erosion(error_volume)
        error_volume = 1.0 * binary_erosion(error_volume)
        nib.save(nib.Nifti1Image(3*error_volume, GT.affine), data_dir+'/error_vol3.nii.gz')
        error_volume = np.zeros_like(gt)
        error_volume[np.where(eval4.get_error() == 1)] = 1
        error_volume = 1.0 * binary_erosion(error_volume)
        error_volume = 1.0 * binary_erosion(error_volume)
        nib.save(nib.Nifti1Image(4*error_volume, GT.affine), data_dir+'/error_vol4.nii.gz')
        error_volume = np.zeros_like(gt)
        error_volume[np.where(gt == -1)] = 1
        nib.save(nib.Nifti1Image(5*error_volume, GT.affine), data_dir+'/tumor.nii.gz')'''


        print('____________')

    print('Done')