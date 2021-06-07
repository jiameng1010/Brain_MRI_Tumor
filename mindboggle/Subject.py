import os
import numpy as np
import trimesh
import nibabel as nib
from skimage import measure
from mindboggle.util import parse_id, parse_group, load_gz_file, \
    load_vtk_file
import subprocess
from subprocess import Popen, PIPE

file_list = ['labels.DKT31.manual+aseg.MNI152.nii.gz',  # 0
             'labels.DKT31.manual+aseg.nii.gz',  # 1
             'labels.DKT31.manual.MNI152.nii.gz',  # 2
             'labels.DKT31.manual.nii.gz',  # 3
             't1weighted_brain.MNI152.nii.gz',  # 4
             't1weighted_brain.nii.gz',  # 5
             't1weighted.MNI152.nii.gz',  # 6
             't1weighted.nii.gz', # 7
             't1weighted_brain.MNI152.affine.txt']  # 8

FREESURFER_SBJ_DIR = '/home/mjia/Researches/Volume_Segmentation/subjects/'

def zero_padding(volume):
    zeros = np.zeros(shape=(184, 220, 184))
    zeros[1:-1, 1:-1, 1:-1] = volume
    return zeros

class Subject():
    def __init__(self, mindboggle_data, subject_id):
        self.dataset = mindboggle_data
        self.id = subject_id
        self.group, self.index = parse_id(self.id)
        self.data_group = parse_group(self.group)
        volume_file = self.get_volume_data(4)
        self.affine = volume_file.affine

    def get_volume_data(self, file_index):
        file_name = self.get_volume_data_file(file_index)
        return load_gz_file(file_name)

    def get_iso_surface(self, file_index):
        volume = self.get_volume_data(file_index)
        volume = volume.get_fdata()
        volume[np.where(volume!=0)] = 1
        t1_volume = zero_padding(volume)
        verts, faces = measure.marching_cubes_classic(t1_volume, 0.4, gradient_direction='ascent')
        return trimesh.Trimesh(verts-1, faces)


    def get_affine_MNI152(self):
        affine_file = self.get_volume_data_file(8)
        return np.loadtxt(affine_file)

    def get_affine(self):
        from util_package.util import load_affine_transform
        supposed_FS_reg_dir = FREESURFER_SBJ_DIR + self.id + '/mri/transforms/'
        supposed_FS_reg_file = supposed_FS_reg_dir + 'talairach.lta'
        if os.path.isfile(supposed_FS_reg_file):
            affine_talairach = load_affine_transform(supposed_FS_reg_file, start_from=9)
            freesurfer_orig = nib.load(FREESURFER_SBJ_DIR + self.id + '/mri/orig.mgz' )
            affine_orig = freesurfer_orig.affine
            return affine_talairach.dot(np.linalg.inv(affine_orig)).dot(self.affine)
        else:
            cmd = ['-subjid', self.id,
                            '-i', self.get_volume_data_file(4),
                            '-autorecon1', '-gcareg']
            command = 'recon-all'
            for i in cmd:
                command = command+' '+i
            return command+'\n'



    def get_volume_data_file(self, file_index):
        supposed_volume_data_dir = os.path.join(os.path.join(self.dataset.data_path_brains), self.data_group + '_volumes')
        if not os.path.isdir(supposed_volume_data_dir):
            self.dataset.unzip_volume_file(self.data_group)
        supposed_volume_data_dir = os.path.join(supposed_volume_data_dir, self.id)
        if not os.path.isdir(supposed_volume_data_dir):
            print('error')
        file_name = file_list[file_index]
        file_name = supposed_volume_data_dir + '/' + file_name
        return file_name

    def get_surface_data(self, file_index):
        file_list = ['lh.labels.DKT25.manual.vtk',  # 0
                     'lh.labels.DKT31.manual.vtk',  # 1
                     'rh.labels.DKT25.manual.vtk',  # 2
                     'rh.labels.DKT31.manual.vtk']  # 3
        supposed_surface_data_dir = os.path.join(self.dataset.data_path_brains, self.data_group+'_surfaces')
        if not os.path.isdir(supposed_surface_data_dir):
            self.dataset.unzip_surface_file(self.data_group)
        supposed_surface_data_dir = os.path.join(supposed_surface_data_dir, self.id)
        if not os.path.isdir(supposed_surface_data_dir):
            print('error')
        file_name = file_list[file_index]
        file_name = supposed_surface_data_dir + '/' + file_name
        return load_vtk_file(file_name)

    '/home/mjia/Downloads/mindboggle/mindboggle_manually_labeled_individual_brains/NKI-RS-22_volumes/NKI-RS-22-10 '