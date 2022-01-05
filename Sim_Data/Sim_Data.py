from medpy.io import load
import numpy as np
import trimesh
from mindboggle.util import load_vtk_file
from util_package import util, plot, constants
from Working_Environment.environment_variables import *

file_list = ['t1.mha',  # 0
             'p_vessel.mha',  # 1
             'p_gray.mha',  # 2
             'p_white.mha', # 3
             'p_csf.mha',  # 4
             'dti.mha',  # 5
             'labels.mha', # 6
             'mesh.vtk']  # 7

def zero_padding(volume):
    zeros = np.zeros(shape=(258, 258, 183))
    zeros[1:-1, 1:-1, 1:-1] = volume
    return zeros

class Sim_Data:
    def __init__(self, dataset_path, data_id):
        self.data_path = dataset_path + "/TumorSimInput" + str(data_id)
        self.data_id = data_id
        _, image_header = load(self.data_path + '/' + file_list[0])
        self.affine = constants.BrainSim_Affine

    def get_volume_data(self, file_index=0):
        volume, image_header = load(self.data_path + '/' + file_list[file_index])
        from medpy.io import save
        save(volume, self.data_path + '/t1.nii', hdr=image_header)
        return volume

    def get_fs_t1file(self):
        return FREESURFER_SBJ_DIR + 'TumorSim_' + str(self.data_id) + '/mri/T1.mgz'

    def get_volume_data_file(self, file_index=0):
        return self.data_path + '/' + file_list[file_index]

    def get_iso_surface(self, file_index=0):
        from skimage import measure
        volume, image_header = load(self.data_path + '/' + file_list[file_index])
        volume[np.where(volume!=0)] = 1
        t1_volume = zero_padding(volume)
        verts, faces = measure.marching_cubes_classic(t1_volume, 0.4, gradient_direction='ascent')
        return trimesh.Trimesh(verts-1, faces)

    def get_mesh(self):
        load_vtk_file(self.data_path + '/' + file_list[7])

    def get_affine(self):
        import os
        import nibabel as nib
        from util_package.util import load_affine_transform
        supposed_FS_reg_dir = FREESURFER_SBJ_DIR + 'TumorSim_' + str(self.data_id) + '/mri/transforms/'
        supposed_FS_reg_file = supposed_FS_reg_dir + 'talairach.lta'
        if os.path.isfile(supposed_FS_reg_file):
            affine_talairach = load_affine_transform(supposed_FS_reg_file, start_from=9)
            freesurfer_orig = nib.load(FREESURFER_SBJ_DIR + 'TumorSim_' + str(self.data_id) + '/mri/orig.mgz' )
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
