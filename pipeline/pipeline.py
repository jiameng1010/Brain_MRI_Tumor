#nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/BrainTumor_test/test_BraTS2020_164/ -o OUTPUT_DIRECTORY -t 1 -m 3d_fullres
import voxelmorph as vxm
import os, sys, copy
import subprocess
import argparse
import nibabel as nib
import h5py
from medpy.io import load, save
import numpy as np

PROJECT_BASE_DIR = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code'
sys.path.append(PROJECT_BASE_DIR)
from Working_Environment.environment_variables import *
from util_package import constants

DISP_FUN_DIR = PCNN_BARIN_DISP_DIR + '/normal_prediction/Displacement_Function'
sys.path.append(DISP_FUN_DIR)

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

class pipeline():
    def __init__(self, inputdir, output_folder, num_points, is_synthesised,
                 resume=False, use_BraTS_label=False,
                 disp_scale=0.10):
        self.input_dir = inputdir
        self.output_dir = output_folder
        self.num_points = num_points
        self.reg_atlas_dir = inputdir + '/reg_atlse/'
        self.input_shape = None
        self.is_synthesised = is_synthesised
        self.disp_scale = disp_scale
        self.use_BraTS_label = use_BraTS_label
        if not resume:
            if os.path.isdir(self.output_dir):
                subprocess.run(['rm', '-r', self.output_dir])
            subprocess.run(['mkdir', self.output_dir])
            if os.path.isdir(self.reg_atlas_dir):
                subprocess.run(['rm', '-r', self.reg_atlas_dir])
            subprocess.run(['mkdir', self.reg_atlas_dir])
        if not os.path.isdir(self.output_dir):
            subprocess.run(['mkdir', self.output_dir])
        for item in os.scandir(inputdir):
            if item.path.endswith('_0001.nii.gz'):
                self.t1 = item.path
                self.data_id = item.path.split("/")[-1][:-12]
                break
        self.freesurfer_dir = self.input_dir + '/' + self.data_id
        self.fs_dir1 = self.input_dir + '/' + self.data_id
        self.fs_dir2 = self.input_dir + '/' + self.data_id + 'seg'

    def get_input_shape(self):
        if self.input_shape == None:
            input, _ = load(self.t1)
            self.input_shape = list(input.shape)
            return copy.copy(self.input_shape)
        else:
            return copy.copy(self.input_shape)

    def get_input_affine(self):
        org_img = nib.load(self.t1)
        return org_img.affine

    #input files -> ./tumor_segmentation/[input_file_name].nii.gz
    def tumor_segmentation(self):
        if self.is_synthesised:
            labels, _ = load(self.input_dir + '/SimTumor_warped_labels2.mha')
            labels_output = np.zeros_like(labels)

            edma_prob, _ = load(self.input_dir + '/SimTumor_prob4.mha')
            core_prob, _ = load(self.input_dir + '/SimTumor_prob5.mha')
            import scipy.ndimage
            edma_vol = np.zeros_like(labels)
            edma_vol[np.where(((edma_prob + core_prob) / 65536) >= 0.5)] = 1
            edma_vol = 1 * scipy.ndimage.binary_closing(edma_vol, structure=np.ones((10, 10, 10)))
            edma_vol = 1 * scipy.ndimage.binary_fill_holes(edma_vol)
            tumor_vol = np.zeros_like(labels)
            tumor_vol[np.where((core_prob / 65536) >= 0.5)] = 1
            tumor_vol = 1 * scipy.ndimage.binary_fill_holes(tumor_vol)

            labels_output[np.where(edma_vol == 1)] = 1
            labels_output[np.where(tumor_vol == 1)] = 3
            subprocess.run(['mkdir', self.input_dir+'/tumor_segmentation'])
            save(labels_output, self.input_dir+'/tumor_segmentation/'+self.data_id+'.nii.gz')
        elif self.use_BraTS_label:
            subprocess.run(['mkdir', self.input_dir + '/tumor_segmentation'])
            subprocess.run(['mv', self.input_dir+self.data_id+'_seg.nii.gz', self.input_dir+'/tumor_segmentation/'+self.data_id+'.nii.gz'])
        else:
            subprocess.run(['mkdir', self.input_dir+'/org'])
            subprocess.run(['cp', self.input_dir + '/' + self.data_id + '_0000.nii.gz', self.input_dir + '/org'])
            subprocess.run(['cp', self.input_dir + '/' + self.data_id + '_0001.nii.gz', self.input_dir + '/org'])
            subprocess.run(['cp', self.input_dir + '/' + self.data_id + '_0002.nii.gz', self.input_dir + '/org'])
            subprocess.run(['cp', self.input_dir + '/' + self.data_id + '_0003.nii.gz', self.input_dir + '/org'])
            command = ['nnUNet_predict',
                       '-i', self.input_dir+'/org',
                       '-o', self.input_dir+'/tumor_segmentation',
                       '-t', '1', '-m', '3d_fullres', '--overwrite_existing']
            env = os.environ.copy()
            env['nnUNet_raw_data_base'] = "/home/mjia/Researches/Volume_Segmentation/nnUNet/nnUNet_raw_data_base"
            env['nnUNet_preprocessed'] = '/home/mjia/Researches/Volume_Segmentation/nnUNet/nnUNet_preprocessed'
            env['RESULTS_FOLDER'] = '/home/mjia/Researches/Volume_Segmentation/nnUNet/RESULTS_FOLDER'
            #subprocess.run(['mkdir', self.input_dir+'/tumor_segmentation'])
            subprocess.run(command, env=env)

        for item in os.scandir(self.input_dir+'/tumor_segmentation'):
            if item.path.endswith('.nii.gz'):
                return item.path

    # input t1 file -> ./[DATA_ID]/mri/transforms/talairach.lta
    # and if(auto_aseg=True) ./[daat_id]/mri/aseg.auto.mgz, aseg.auto_noCCseg.mgz, aseg.presurf.mgz
    def talairach_registration(self, auto_aseg=False):
        if os.path.isdir(self.input_dir + self.data_id):
            subprocess.run(['rm', '-r', self.input_dir + self.data_id])
        if auto_aseg:
            command = ['recon-all',
                       '-subjid', self.data_id,
                       '-i', self.t1,
                       '-autorecon1', '-gcareg', '-canorm', '-careg', '-rmneck', '-skull-lta', '-calabel']
        else:
            command = ['recon-all',
                       '-subjid', self.data_id,
                       '-i', self.t1,
                       '-autorecon1', '-gcareg']

        subprocess.run(['rm', 'FreeSurfer_script.sh'])
        with open('FreeSurfer_script.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('export FREESURFER_HOME=' + FREESURFER_HOME + '\n')
            f.write('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
            f.write('export SUBJECTS_DIR=' + self.input_dir + '\n')
            f.write(" ".join(command))

        subprocess.run(['chmod', '777', './FreeSurfer_script.sh'])
        subprocess.check_output(['./FreeSurfer_script.sh'])
        return self.input_dir + self.data_id +'/mri/transforms/talairach.lta'

    # ./tumor_segmentation/[input_file_name].nii.gz
    # +
    # ./[self.data_id]/mri/transforms/talairach.lta
    # -> Tumor_Point_Cloud.h5
    def produce_point_cloud(self):
        import numpy as np
        import nibabel as nib
        import scipy, trimesh, skimage.measure
        from medpy.io import load, save
        from util_package.util import load_affine_transform, zero_padding, affine
        from Training_Data_Generator.Training_Data_Generator import volume_to_mesh, volume_to_mesh_continus

        tumor_volume, _ = load(self.input_dir+'/tumor_segmentation/'+self.data_id+'.nii.gz')
        #tumor_volume[np.where(tumor_volume != 3)] = 0  #
        seg_volume = np.zeros_like(tumor_volume)
        seg_volume[np.where(tumor_volume >= 1)] = 1  # edma
        edma_mesh = volume_to_mesh(seg_volume)
        seg_volume[np.where(tumor_volume == 1)] = 0  # edma
        tumor_mesh = volume_to_mesh(seg_volume)

        points, _ = trimesh.sample.sample_surface(tumor_mesh, self.num_points)
        edma_points, _ = trimesh.sample.sample_surface(edma_mesh, self.num_points)

        if self.is_synthesised:
            from Sim_Data.Sim_Data import Sim_Data
            from util_package.util import parse_source_info
            source = parse_source_info(self.input_dir + '/Source_info')
            BrainSim_inputdata = Sim_Data(BrainSim_inputdata_dir, source[0])
            affine_matrix = BrainSim_inputdata.get_affine()
        else:
            affine_talairach = load_affine_transform(self.freesurfer_dir + '/mri/transforms/talairach.lta', start_from=9)
            freesurfer_orig = nib.load(self.freesurfer_dir + '/mri/orig.mgz')
            affine_orig = freesurfer_orig.affine
            _, affine_t1 = vxm.py.utils.load_volfile(self.t1, ret_affine=True)
            affine_matrix = affine_talairach.dot(np.linalg.inv(affine_orig)).dot(affine_t1)

        points_transformed = affine(affine_matrix, points)
        points_transformed = affine(np.linalg.inv(constants.Affine_uniform2talairach), points_transformed)
        points_edma_transformed = affine(affine_matrix, edma_points)
        points_edma_transformed = affine(np.linalg.inv(constants.Affine_uniform2talairach), points_edma_transformed)
        with h5py.File(self.output_dir + '/pointcloud.h5', 'w') as f:
            f.create_dataset('pointcloud_tumor', data=points_transformed)
            f.create_dataset('pointcloud_edma', data=points_edma_transformed)
        return points_transformed

    def pred_svf(self):
        from DispFunction import DispFunction
        import nibabel as nib
        import numpy as np
        PC_f = h5py.File(self.output_dir + '/pointcloud.h5', 'r')
        pointcloud = PC_f['pointcloud_tumor']
        pc_extend = np.asarray([np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]),
                    np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
                    np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])

        trained_model_dir = PCNN_BARIN_DISP_DIR + '/normal_prediction/train_results/' + '2021_10_18_18_38_55' + '/trained_models/model.ckpt'
        df = DispFunction(trained_model_dir)
        df.update_PC(pointcloud, pc_extend)

        from util_package.util import load_affine_transform
        affine_talairach = load_affine_transform(self.freesurfer_dir + '/mri/transforms/talairach.lta', start_from=9)
        freesurfer_orig = nib.load(self.freesurfer_dir + '/mri/orig.mgz')
        affine_orig = freesurfer_orig.affine
        _, affine = vxm.py.utils.load_volfile(self.t1, ret_affine=True)
        svf = df.get_displacement_image(affine_to_talairach=np.linalg.inv(constants.Affine_uniform2talairach).dot(affine_talairach.dot(np.linalg.inv(affine_orig)).dot(affine)),
                                        shape=[240, 240, 155])
        save(svf, self.output_dir + '/SVF.mha')
        return self.output_dir + '/SVF.mha'

    def pred_disp(self):
        from DispFunction import DispFunction
        import nibabel as nib
        import numpy as np
        PC_f = h5py.File(self.output_dir + '/pointcloud.h5', 'r')
        pointcloud = PC_f['pointcloud_tumor']
        pc_extend = np.asarray([np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]),
                    np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
                    np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])

        trained_model_dir = PCNN_BARIN_DISP_DIR + '/normal_prediction/train_results/' + '2021_10_18_18_38_55' + '/trained_models/model.ckpt'
        df = DispFunction(trained_model_dir)
        df.update_PC(pointcloud, pc_extend)

        from util_package.util import load_affine_transform, rotation
        affine_talairach = load_affine_transform(self.freesurfer_dir + '/mri/transforms/talairach.lta', start_from=9)
        freesurfer_orig = nib.load(self.freesurfer_dir + '/mri/orig.mgz')
        affine_orig = freesurfer_orig.affine
        _, affine_t1 = vxm.py.utils.load_volfile(self.t1, ret_affine=True)
        svf = df.get_displacement_image(affine_to_talairach=np.linalg.inv(constants.Affine_uniform2talairach).dot(affine_talairach.dot(np.linalg.inv(affine_orig)).dot(affine_t1)),
                                        shape=self.get_input_shape())

        save(svf, self.output_dir + '/displacement_pred.mha')
        return self.output_dir + '/displacement_pred.mha'

    def invert_disp(self):
        #import SimpleITK as sitk
        #displacement = self.load_pred_disp()
        #displacement = sitk.GetImageFromArray(displacement, isVector=True)
        #inv_disp = sitk.InverseDisplacementField(displacement)
        displacement = self.load_pred_disp()
        save(displacement, self.output_dir + '/displacement_scale.mha')

        subprocess.run([MIRTK_EXECUTABLE,
                        'convert-dof',
                        self.output_dir + '/displacement_scale.mha',
                        self.output_dir + '/displacement',
                        '-input-format', 'disp_world',
                        '-output-format', 'mirtk_bspline_ffd'])
        subprocess.run([MIRTK_EXECUTABLE,
                        'invert-dof',
                        self.output_dir + '/displacement',
                        self.output_dir + '/displacement_inv'])
        subprocess.run([MIRTK_EXECUTABLE,
                        'convert-dof',
                        self.output_dir + '/displacement_inv',
                        self.output_dir + '/displacement_inv.mha',
                        '-format', 'disp_world'])

        return self.output_dir + '/displacement_inv.mha'


    def svf_to_displacement(self):
        svf, _ = load(self.output_dir + '/SVF.mha')
        svf = -self.disp_scale * svf
        save(svf, self.output_dir + '/SVF_modified.mha')
        svf_file = self.output_dir + '/SVF_modified.mha'
        subprocess.run([MIRTK_EXECUTABLE,
                        'calculate-exponential-map',
                        svf_file,
                        self.output_dir + '/displacement.mha'])
        return self.output_dir + '/displacement.mha'

    def load_pred_disp(self, is_inv=False):
        from util_package.util import interpolate, rotation, load_affine_transform
        import nibabel as nib
        if is_inv:
            if os.path.isfile(self.output_dir + '/displacement_inv.mha'):
                displacement, _ = load(self.output_dir + '/displacement_inv.mha')
            return displacement
        else:
            displacement, _ = load(self.output_dir + '/displacement_pred.mha')
            displacement = np.reshape(displacement, [-1, 3])
            affine_talairach = load_affine_transform(self.freesurfer_dir + '/mri/transforms/talairach.lta', start_from=9)
            freesurfer_orig = nib.load(self.freesurfer_dir + '/mri/orig.mgz')
            affine_orig = freesurfer_orig.affine
            _, affine = vxm.py.utils.load_volfile(self.t1, ret_affine=True)
            displacement = rotation(np.linalg.inv(np.linalg.inv(constants.Affine_uniform2talairach).dot(
                affine_talairach.dot(np.linalg.inv(affine_orig)).dot(affine))), displacement)
            input_shape = self.get_input_shape()
            input_shape.append(3)
            displacement = np.reshape(displacement, input_shape)
            return self.disp_scale * displacement

    def apply_displacement(self):
        from util_package.util import interpolate, rotation, load_affine_transform
        t1, _ = load(self.t1)
        displacement = self.load_pred_disp()
        wapped_t1 = interpolate(t1, displacement)
        import nibabel as nib
        org_image = nib.load(self.t1)
        wapped_image = nib.Nifti1Image(wapped_t1, org_image.affine)
        nib.save(wapped_image, self.output_dir + '/wapped_t1.nii.gz')

        displacement = self.load_pred_disp(is_inv=True)
        unwapped_t1 = interpolate(wapped_t1, displacement)
        unwapped_image = nib.Nifti1Image(unwapped_t1, org_image.affine)
        nib.save(unwapped_image, self.output_dir + '/unwapped_t1.nii.gz')
        return self.output_dir + '/wapped_t1.nii.gz'

    def reg_atlas(self, src, dist):
        if not os.path.isdir(self.reg_atlas_dir):
            subprocess.run(['mkdir', self.reg_atlas_dir])

        img_atlas = nib.load(ATROPOS_TEMPLATE)
        img = nib.load(src)
        img_out = nib.Nifti1Image(img.get_fdata(), img_atlas.affine)
        nib.save(img_out, self.reg_atlas_dir + '/T1wapped.nii.gz')
        subprocess.run([MIRTK_EXECUTABLE, 'register',
                        self.reg_atlas_dir + '/T1wapped.nii.gz',
                        ATROPOS_TEMPLATE,
                        '-dofout', self.reg_atlas_dir + '/deformation_atlas',
                        '-output', self.reg_atlas_dir + '/registered_t1.nii.gz',
                        '-be', '0.0001'])
        subprocess.run([MIRTK_EXECUTABLE,
                        'invert-dof',
                        self.reg_atlas_dir + '/deformation_atlas',
                        self.reg_atlas_dir + '/deformation_atlas_inv'])
        subprocess.run([MIRTK_EXECUTABLE, 'transform-image',
                        ATROPOS_TEMPLATE_SEG1, self.reg_atlas_dir + '/seg_atlas.nii.gz',
                        '-dofin', self.reg_atlas_dir + '/deformation_atlas',
                        '-interp', 'NN'])
        '''subprocess.run([MIRTK_EXECUTABLE, 'convert-dof',
                        self.reg_atlas_dir + '/deformation_atlas',
                        self.reg_atlas_dir + '/deformation_atlas.nii.gz',
                        '-format', 'disp_world'])
        subprocess.run([MIRTK_EXECUTABLE, 'resample-image',
                        self.reg_atlas_dir + '/deformation_atlas',
                        self.reg_atlas_dir + '/deformation_atlas.nii.gz',
                        '-size', '1', '1', '1',
                        '-interp', 'Linear'])
        subprocess.run([MIRTK_EXECUTABLE, 'transform-image',
                        ATROPOS_TEMPLATE_SEG2, self.reg_atlas_dir + '/seg_atlas.nii.gz',
                        '-dofin', self.reg_atlas_dir + '/deformation_atlas'])'''
        seg_volume = nib.load(self.reg_atlas_dir + '/seg_atlas.nii.gz')
        img_out = nib.Nifti1Image(seg_volume.get_fdata(), img.affine)
        nib.save(img_out, dist)
        return dist

    def segmentation_freesurfer(self):
        if os.path.isdir(self.input_dir + self.data_id + 'seg'):
            subprocess.run(['rm', '-r', self.input_dir + self.data_id + 'seg'])
        command = ['recon-all',
                   '-subjid', self.data_id + 'seg',
                   '-i', self.output_dir + '/wapped_t1.nii.gz',
                   '-autorecon1', '-gcareg', '-canorm', '-careg', '-rmneck', '-skull-lta', '-calabel']

        subprocess.run(['rm', 'FreeSurfer_script.sh'])
        with open('FreeSurfer_script.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('export FREESURFER_HOME=' + FREESURFER_HOME + '\n')
            f.write('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
            f.write('export SUBJECTS_DIR=' + self.input_dir + '\n')
            f.write(" ".join(command))

        subprocess.run(['chmod', '777', './FreeSurfer_script.sh'])
        subprocess.check_output(['./FreeSurfer_script.sh'])
        return self.input_dir + self.data_id + 'seg'

    def project_samseg(self, src, dist, direct_copy=False):
        if direct_copy:
            seg = nib.load(src)
            nib.save(seg, dist)
            return
        seg = nib.load(src)
        seg_img = np.squeeze(seg.get_fdata())
        seg_affine = seg.affine
        displacement = self.load_pred_disp(is_inv=True)

        output_image = np.zeros_like(seg_img)
        index = np.where(output_image == 0)
        voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                             np.expand_dims(index[1], axis=1),
                                             np.expand_dims(index[2], axis=1)], axis=1)
        voxels_affine = voxels_interpolate + np.reshape(displacement, (-1, 3))
        shape = seg_img.shape
        x = np.linspace(0, shape[0] - 1, shape[0])
        y = np.linspace(0, shape[1] - 1, shape[1])
        z = np.linspace(0, shape[2] - 1, shape[2])
        from scipy.interpolate import RegularGridInterpolator
        from util_package.util import clap_voxels_out
        fn = RegularGridInterpolator((x, y, z), seg_img, method='nearest')
        voxels_in_displacement, index = clap_voxels_out(voxels_affine, index, shape)
        output = fn(voxels_in_displacement)
        output_image[index] = output
        unwapped_seg = nib.Nifti1Image(output_image, self.get_input_affine())
        nib.save(unwapped_seg, dist)

    def project_segment(self, src, dist):
        seg = nib.load(src)
        seg_img = np.squeeze(seg.get_fdata())
        seg_affine = seg.affine
        org_affine = self.get_input_affine()
        segmentation = segmentation_geometric_transformation(seg_img, np.linalg.inv(seg_affine).dot(org_affine), self.get_input_shape())

        unwapped_seg = nib.Nifti1Image(segmentation, self.get_input_affine())
        nib.save(unwapped_seg, dist)

    def project_segment_back(self, src, dist):
        seg = nib.load(src)
        seg_img = np.squeeze(seg.get_fdata())
        seg_affine = seg.affine
        org_affine = self.get_input_affine()
        segmentation = segmentation_geometric_transformation(seg_img, np.linalg.inv(seg_affine).dot(org_affine), self.get_input_shape())
        displacement = self.load_pred_disp(is_inv=True)

        output_image = np.zeros_like(segmentation)
        index = np.where(output_image == 0)
        voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                             np.expand_dims(index[1], axis=1),
                                             np.expand_dims(index[2], axis=1)], axis=1)
        voxels_affine = voxels_interpolate + np.reshape(displacement, (-1, 3))
        shape = segmentation.shape
        x = np.linspace(0, shape[0] - 1, shape[0])
        y = np.linspace(0, shape[1] - 1, shape[1])
        z = np.linspace(0, shape[2] - 1, shape[2])
        from scipy.interpolate import RegularGridInterpolator
        from util_package.util import clap_voxels_out
        fn = RegularGridInterpolator((x, y, z), segmentation, method='nearest')
        voxels_in_displacement, index = clap_voxels_out(voxels_affine, index, shape)
        output = fn(voxels_in_displacement)
        output_image[index] = output
        unwapped_seg = nib.Nifti1Image(output_image, self.get_input_affine())
        nib.save(unwapped_seg, dist)

    def segmentation_samseg_org(self, input_file, output_folder):
        command = ['run_samseg',
                   '-o', output_folder,
                   '-i', input_file,
                   #'--lesion',
                   #'--lesion-mask-pattern', '1',
                   '--threads', '4']

        subprocess.run(['rm', 'FreeSurfer_script.sh'])
        with open('FreeSurfer_script.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('export FREESURFER_HOME=' + FREESURFER_HOME + '\n')
            f.write('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
            f.write('export SUBJECTS_DIR=' + self.input_dir + '\n')
            f.write(" ".join(command))

        subprocess.run(['chmod', '777', './FreeSurfer_script.sh'])
        subprocess.check_output(['./FreeSurfer_script.sh'])

    def segmentation_samseg(self, input_file, output_folder):
        sys.path.append(SAMSEG_DIR)
        from freesurfer.samseg import Samseg, ProbabilisticAtlas
        ss = Samseg(imageFileNames=[input_file],
                    atlasDir=SAMSEG_ATLAS_DIR1,
                    savePath=output_folder,)
                    #imageToImageTransformMatrix=affine_v2v)
        ss.segment()

    def samseg_def(self, input_file, output_folder, use_GT_disp=False):
        sys.path.append(SAMSEG_DIR)
        from freesurfer.samseg import Samseg, ProbabilisticAtlas, SamsegTumor

        if os.path.isdir(output_folder):
            subprocess.run(['rm', output_folder])
        subprocess.run(['mkdir', output_folder])
        subprocess.run(['mkdir', output_folder + '/atlas'])

        subprocess.run(['cp', SAMSEG_ATLAS_DIR1 + '/compressionLookupTable.txt', output_folder + '/atlas'])
        subprocess.run(['cp', SAMSEG_ATLAS_DIR1 + '/sharedGMMParameters.txt', output_folder + '/atlas'])
        subprocess.run(['cp', SAMSEG_ATLAS_DIR1 + '/template.nii', output_folder + '/atlas'])

        if use_GT_disp:
            displacement, _ = load(self.input_dir + '/SimTumor_def_inverse.mha')
            displacement = self.disp_scale * displacement
            displacement_inv, _ = load(self.input_dir + '/SimTumor_def.mha')
            displacement_inv = self.disp_scale * displacement_inv
        else:
            displacement = self.load_pred_disp()
            displacement_inv = self.load_pred_disp(is_inv=True)
        from util_package.util import load_affine_transform_true, affine, clap_voxels_out, rotation
        from scipy.interpolate import RegularGridInterpolator
        affine1 = nib.load(self.t1).affine
        affine2 = load_affine_transform_true(self.input_dir + '/samseg1/template.lta', 9)
        affine3 = nib.load(output_folder + '/atlas/template.nii').affine
        affine_v2v = np.linalg.inv(affine1).dot(affine2.dot(affine3))
        if use_GT_disp:
            from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
            generator = Training_Data_Generator(None, None, None, self.input_dir, None)
            affine4 = generator.BrainSim.get_affine()
            affine5 = generator.Mindboggle_data.get_affine()
            affine_v2vd = np.linalg.inv(affine4).dot(affine5).dot(affine_v2v)
        else:
            affine_v2vd = affine_v2v
        def deform_atlas(atlas_file):
            probabilisticAtlas = ProbabilisticAtlas()
            mesh = probabilisticAtlas.getMesh(atlas_file)

            from util_package.util import clap_voxels_out2
            shape = displacement.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), displacement, fill_value=0.0)

            disp = np.zeros_like(mesh.points)
            transformed_vertices = affine(affine_v2vd, mesh.points)
            transformed_vertices, index = clap_voxels_out2(transformed_vertices, np.arange(disp.shape[0]), shape)
            disp[index, :] = fn(transformed_vertices)
            disp = rotation(np.linalg.inv(affine_v2vd), disp)
            probabilisticAtlas.saveDeformedAtlas(atlas_file,
                                                 output_folder + '/atlas/' + atlas_file.split('/')[-1][:-3],
                                                 mesh.points + disp)

        def deform_atlas_vol(volume_atlas_file):
            vol_atlas = nib.load(volume_atlas_file).get_fdata()
            index = np.where(vol_atlas >= 0)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_in_displacement = affine(affine_v2vd, voxels_interpolate)
            shape = displacement_inv.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), displacement_inv)
            voxels_in_displacement, index = clap_voxels_out(voxels_in_displacement, index, shape)
            disp = fn(voxels_in_displacement)
            disp_rot = rotation(np.linalg.inv(affine_v2vd), disp)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_in_displacement_moved = voxels_interpolate + disp_rot

            shape = vol_atlas.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), vol_atlas)
            voxels_in_atlas, index = clap_voxels_out(voxels_in_displacement_moved, index, shape)
            output = np.zeros_like(vol_atlas)
            output[index] = fn(voxels_in_atlas)
            nib.save(nib.Nifti1Image(output, nib.load(volume_atlas_file).affine), output_folder + '/atlas/' + volume_atlas_file.split('/')[-1])


        deform_atlas_vol(SAMSEG_ATLAS_DIR1 + '/template.nii')
        deform_atlas(SAMSEG_ATLAS_DIR1 + '/atlasForAffineRegistration.txt.gz')
        deform_atlas(SAMSEG_ATLAS_DIR1 + '/atlas_level1.txt.gz')
        deform_atlas(SAMSEG_ATLAS_DIR1 + '/atlas_level2.txt.gz')

        probabilisticAtlas = ProbabilisticAtlas()
        mesh = probabilisticAtlas.getMesh(output_folder + '/atlas/atlas_level2.txt.gz')
        img = mesh.rasterize(nib.load(output_folder + '/atlas/template.nii').get_fdata().shape, 4).astype(float)
        nib.save(nib.Nifti1Image(img, nib.load(output_folder + '/atlas/template.nii').affine), output_folder + '/atlas/test_vol_atlas.nii')

        ss = SamsegTumor(imageFileNames=[input_file],
                    atlasDir=output_folder+'/atlas',
                    savePath=output_folder,)
                    #imageToImageTransformMatrix=affine_v2v)
        ss.segment()
        pass

    def samseg_tumor_def(self, input_file, output_folder, use_GT_disp=False):
        sys.path.append(SAMSEG_DIR)
        from freesurfer.samseg import Samseg, ProbabilisticAtlas, SamsegTumor
        if os.path.isdir(output_folder):
            subprocess.run(['rm', '-r', output_folder])
        subprocess.run(['mkdir', output_folder])

        subprocess.run(['cp', '-r', SAMSEG_ATLAS_DIR1, output_folder])
        subprocess.run(['mv', '-r', output_folder + '/20Subjects_smoothing2_down2_smoothingForAffine2', output_folder + '/atlas'])

        if use_GT_disp:
            displacement, _ = load(self.input_dir + '/SimTumor_def_inverse.mha')
            displacement = self.disp_scale * displacement
            displacement_inv, _ = load(self.input_dir + '/SimTumor_def.mha')
            displacement_inv = self.disp_scale * displacement_inv
        else:
            displacement = self.load_pred_disp()
            displacement_inv = self.load_pred_disp(is_inv=True)
        from util_package.util import load_affine_transform_true, affine, clap_voxels_out, rotation
        from scipy.interpolate import RegularGridInterpolator
        affine1 = nib.load(self.t1).affine
        affine2 = load_affine_transform_true(self.input_dir + '/samseg1/template.lta', 9)
        affine3 = nib.load(output_folder + '/atlas/template.nii').affine
        affine_v2v = np.linalg.inv(affine1).dot(affine2.dot(affine3))
        if use_GT_disp:
            from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
            generator = Training_Data_Generator(None, None, None, self.input_dir, None)
            affine4 = generator.BrainSim.get_affine()
            affine5 = generator.Mindboggle_data.get_affine()
            affine_v2vd = np.linalg.inv(affine4).dot(affine5).dot(affine_v2v)
        else:
            affine_v2vd = affine_v2v
        def deform_atlas(atlas_file):
            probabilisticAtlas = ProbabilisticAtlas()
            mesh = probabilisticAtlas.getMesh(atlas_file)

            from util_package.util import clap_voxels_out2
            shape = displacement.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), displacement, fill_value=0.0)

            disp = np.zeros_like(mesh.points)
            transformed_vertices = affine(affine_v2vd, mesh.points)
            transformed_vertices, index = clap_voxels_out2(transformed_vertices, np.arange(disp.shape[0]), shape)
            disp[index, :] = fn(transformed_vertices)
            disp = rotation(np.linalg.inv(affine_v2vd), disp)
            probabilisticAtlas.saveDeformedAtlas(atlas_file,
                                                 output_folder + '/atlas/' + atlas_file.split('/')[-1][:-3],
                                                 mesh.points + disp)

        def deform_atlas_vol(volume_atlas_file):
            vol_atlas = nib.load(volume_atlas_file).get_fdata()
            index = np.where(vol_atlas >= 0)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_in_displacement = affine(affine_v2vd, voxels_interpolate)
            shape = displacement_inv.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), displacement_inv)
            voxels_in_displacement, index = clap_voxels_out(voxels_in_displacement, index, shape)
            disp = fn(voxels_in_displacement)
            disp_rot = rotation(np.linalg.inv(affine_v2vd), disp)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_in_displacement_moved = voxels_interpolate + disp_rot

            shape = vol_atlas.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), vol_atlas)
            voxels_in_atlas, index = clap_voxels_out(voxels_in_displacement_moved, index, shape)
            output = np.zeros_like(vol_atlas)
            output[index] = fn(voxels_in_atlas)
            nib.save(nib.Nifti1Image(output, nib.load(volume_atlas_file).affine), output_folder + '/atlas/' + volume_atlas_file.split('/')[-1])

        deform_atlas_vol(SAMSEG_ATLAS_DIR1 + '/template.nii')
        deform_atlas(SAMSEG_ATLAS_DIR1 + '/atlasForAffineRegistration.txt.gz')
        deform_atlas(SAMSEG_ATLAS_DIR1 + '/atlas_level1.txt.gz')
        deform_atlas(SAMSEG_ATLAS_DIR1 + '/atlas_level2.txt.gz')

        SamsegTumor.makeTumorAtlas(output_folder + '/atlas',
                               output_folder + '/atlas_tumor')
        ss = SamsegTumor.SamsegTumor(imageFileNames=[input_file],
                                     atlasDir=output_folder + '/atlas_tumor',
                                     savePath=output_folder,
                                     useMeanConstr=False, useShapeModel=False)
        ss.segment()

    def run(self):
        self.talairach_registration(auto_aseg=True)
        self.project_segment(src=self.fs_dir1 + '/mri/aseg.auto.mgz', dist=self.output_dir + '/fs_seg1.nii.gz')

        self.tumor_segmentation()
        self.produce_point_cloud()
        # self.pred_svf()
        # self.svf_to_displacement()

        self.pred_disp()
        self.invert_disp()
        self.apply_displacement()

        # pp.reg_atlas()

        self.segmentation_freesurfer()
        self.project_segment_back(src=self.fs_dir2+'/mri/aseg.auto.mgz', dist=self.output_dir + '/fs_seg2.nii.gz')

        self.reg_atlas(src=self.input_dir + self.data_id + '/mri/norm.mgz', dist=self.reg_atlas_dir + '/atlas_seg1.nii.gz')
        self.project_segment(src=self.reg_atlas_dir + '/atlas_seg1.nii.gz', dist=self.output_dir + '/atlas_seg1.nii.gz')

        self.reg_atlas(src=self.input_dir + self.data_id + 'seg/mri/norm.mgz', dist=self.reg_atlas_dir + '/atlas_seg2.nii.gz')
        self.project_segment_back(src=self.reg_atlas_dir + '/atlas_seg2.nii.gz', dist=self.output_dir + '/atlas_seg2.nii.gz')

    def run_samseg(self):
        #self.talairach_registration(auto_aseg=False)

        #self.tumor_segmentation()
        #self.produce_point_cloud()
        #self.pred_svf()
        #self.svf_to_displacement()

        #self.pred_disp()
        #return
        #self.invert_disp()
        #self.apply_displacement()

        #self.segmentation_samseg(self.t1, self.input_dir + '/samseg1')
        #self.project_samseg(src=self.input_dir + '/samseg1/seg.mgz', dist=self.output_dir + '/samseg1.nii.gz',
        #                    direct_copy=True)
        #self.segmentation_samseg(self.output_dir+'/wapped_t1.nii.gz', self.input_dir + '/samseg2')
        #self.project_samseg(src=self.input_dir + '/samseg2/seg.mgz', dist=self.output_dir + '/samseg2_' + str(self.disp_scale)[:4] + '.nii.gz')
        self.samseg_tumor_def(self.t1, self.input_dir + '/samseg4', use_GT_disp=False)
        self.project_samseg(src=self.input_dir + '/samseg4/seg.mgz', dist=self.output_dir + '/samseg4_tumor_' + str(self.disp_scale)[:4] + '.nii.gz',
                            direct_copy=True)



    def run_eval(self, pred_file, with_cortex=True):
        from eval import Eval
        pred = nib.load(self.output_dir + '/' + pred_file).get_fdata()
        gt = nib.load(self.input_dir + '/transformed_labels_manual_aseg.nii.gz').get_fdata()
        pred[np.where(gt==-1)] = -1
        rotation = nib.load(self.input_dir + '/transformed_labels_manual_aseg.nii.gz').affine

        '''mask = np.zeros_like(gt)
        mask[np.where(gt==-1)] = 1
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask, structure=np.ones(shape=[5, 5, 5]))
        pred = pred * mask
        gt = gt * mask'''
        #all
        eval1 = Eval(pred, gt, rotation, with_cortex=with_cortex)
        #tumor side
        eval2 = Eval(pred, gt, rotation, tumor_side=True, with_cortex=with_cortex)
        #non-tumor side
        eval3 = Eval(pred, gt, rotation, tumor_side=False, with_cortex=with_cortex)
        return np.asarray([eval1.get_weighted_mean_dice1(), eval2.get_weighted_mean_dice1(), eval3.get_weighted_mean_dice1()])

    def run_eval_org(self, index, with_cortex=True):
        gt_tumor = nib.load(self.input_dir + '/transformed_labels_manual_aseg.nii.gz').get_fdata()

        from eval import Eval
        pred = nib.load(self.output_dir + '/samseg_org.nii.gz').get_fdata()
        gt = nib.load(self.input_dir + '/OASIS-TRT-20-' + str(index) + '_DKT31_CMA_labels.nii.gz').get_fdata()
        pred[np.where(gt==-1)] = -1
        rotation = nib.load(self.input_dir + '/transformed_labels_manual_aseg.nii.gz').affine

        eval_for_tumorside = Eval(pred, gt_tumor, rotation, with_cortex=with_cortex)
        tumor_at_left = eval_for_tumorside.get_tumor_center()
        # all
        eval1 = Eval(pred, gt, rotation, with_cortex=with_cortex, tumor_at_left=tumor_at_left)
        # tumor side
        eval2 = Eval(pred, gt, rotation, tumor_side=True, with_cortex=with_cortex, tumor_at_left=tumor_at_left)
        # non-tumor side
        eval3 = Eval(pred, gt, rotation, tumor_side=False, with_cortex=with_cortex, tumor_at_left=tumor_at_left)
        return np.asarray([eval1.get_weighted_mean_dice1(), eval2.get_weighted_mean_dice1(), eval3.get_weighted_mean_dice1()])

    def plot_result(self, pred_file1, pred_file2, output_file):
        metric1 = self.run_eval(pred_file1)
        metric2 = self.run_eval(pred_file2)
        metric1_nocortex = self.run_eval(pred_file1, with_cortex=False)
        metric2_nocortex = self.run_eval(pred_file2, with_cortex=False)
        from plot_result import Plot_result
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
    # DEFAULT SETTINGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='where input data is')
    parser.add_argument('--output_folder', type=str, help='will create a folder to save final segmentation')
    parser.add_argument('--num_points', type=int, default=1500, help='size of the point cloud, input of the network')
    parser.add_argument('--is_synthesised', type=bool, default=False, help='is the input data synthesised or not')
    parser.add_argument('--disp_scale', type=float, default=0.10, help='the scale factor of the predicted displacement')
    config = parser.parse_args()

    from datetime import datetime
    a = datetime.now()

    pp = pipeline(config.input_folder, config.input_folder+'/'+config.output_folder, config.num_points,
                  is_synthesised=config.is_synthesised, resume=True, disp_scale=config.disp_scale)
    pp.run_samseg()

    '''from eval import Eval
    pred1 = nib.load(pp.output_dir + '/samseg1.nii.gz').get_fdata()
    pred2 = nib.load(pp.output_dir + '/samseg2.nii.gz').get_fdata()
    gt = nib.load(pp.input_dir + '/transformed_labels_manual_aseg.nii.gz').get_fdata()
    eval = Eval(pred1, gt)
    print(eval.get_weighted_mean_dice1())
    eval = Eval(pred2, gt)
    print(eval.get_weighted_mean_dice1())'''

    #pp.talairach_registration(auto_aseg=True)

    '''pp.tumor_segmentation()
    pp.produce_point_cloud()
    #pp.pred_svf()
    #pp.svf_to_displacement()

    pp.pred_disp()
    pp.invert_disp()
    pp.apply_displacement()

    pp.segmentation_freesurfer()
    pp.project_segment()
    pp.project_segment_back()'''

    '''pp.reg_atlas(src=pp.input_dir + pp.data_id + '/mri/T1.mgz', dist=pp.reg_atlas_dir + '/atlas_seg1.nii.gz')
    pp.project_segment(src=pp.reg_atlas_dir + '/atlas_seg1.nii.gz', dist=pp.output_dir + '/atlas_seg1.nii.gz')

    pp.reg_atlas(src=pp.input_dir + pp.data_id + 'seg/mri/T1.mgz', dist=pp.reg_atlas_dir + '/atlas_seg2.nii.gz')
    pp.project_segment_back(src=pp.reg_atlas_dir + '/atlas_seg2.nii.gz', dist=pp.output_dir + '/atlas_seg2.nii.gz')'''

    b= datetime.now()
    print(b-a)
    print('done')

