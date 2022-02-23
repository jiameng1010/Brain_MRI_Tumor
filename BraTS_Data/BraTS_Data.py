import voxelmorph as vxm
import numpy as np
import trimesh
import pymesh
import nibabel as nib
import scipy
from .Tetr_Mesh import Tetr_Mesh
from skimage import measure

FREESURFER_SBJ_DIR = '/home/mjia/Researches/Volume_Segmentation/subjects/'

def clearup_mesh(mesh):
    meshes = trimesh.graph.split(mesh)
    if len(meshes) == 1:
        return meshes[0]
    else:
        outputs = []
        for i in range(len(meshes)):
            if meshes[i].volume > 10:
                outputs.append(meshes[i])
            else:
                print(meshes[i].volume)
                if meshes[i].volume < -50:
                    print(meshes[i].volume)
        return trimesh.util.concatenate(outputs)


def zero_padding(volume):
    zeros = np.zeros(shape=(242, 242, 157))
    zeros[1:-1, 1:-1, 1:-1] = volume
    return zeros


class BraTS_Data:
    def __init__(self, dataset_path, data_id):
        self.data_path = dataset_path + "/BraTS20_Training_" + str(data_id).zfill(3)
        self.data_id = data_id
        self.list_of_files = ['/BraTS20_Training_' + str(data_id).zfill(3) + '_t1.nii.gz',
                              '/BraTS20_Training_' + str(data_id).zfill(3) + '_t1ce.nii.gz',
                              '/BraTS20_Training_' + str(data_id).zfill(3) + '_t2.nii.gz',
                              '/BraTS20_Training_' + str(data_id).zfill(3) + '_flair.nii.gz',
                              '/BraTS20_Training_' + str(data_id).zfill(3) + '_seg.nii.gz']
        _, self.affine = self.produce_data(0, ret_affine=True)

    def get_filename(self, id):
        return self.data_path + self.list_of_files[id]

    def produce_data(self, id, ret_affine=False):
        if isinstance(id, int):
            return vxm.py.utils.load_volfile(self.data_path + self.list_of_files[id], ret_affine=ret_affine)
        else:
            return vxm.py.utils.load_volfile(self.data_path + '/BraTS20_Training_'
                                             + str(self.data_id).zfill(3) + '_' + id + '.nii.gz', ret_affine=ret_affine)

    def produce_undeformed(self, id, save_to=None):
        if isinstance(id, int):
            org_data = vxm.py.utils.load_volfile(self.data_path + self.list_of_files[id])
        else:
            org_data = vxm.py.utils.load_volfile(self.data_path + '/BraTS20_Training_'
                                             + str(self.data_id).zfill(3) + '_' + id + '.nii.gz')
        undeformed = self.tetr_volume.interpolate_displacement(org_data)
        if save_to == None:
            return undeformed
        else:
            data = nib.load(self.data_path + self.list_of_files[id])
            output = nib.Nifti1Image(undeformed, data.affine)
            nib.save(output, save_to+'/Undeformed_'+self.list_of_files[id][1:])
            return undeformed

    def get_affine(self):
        t1_volume, affine = self.produce_data(0, True)
        return affine

    def get_tumor_size(self):
        seg_volume = self.produce_data(4)
        four = len(np.where(seg_volume == 4)[0])
        two = len(np.where(seg_volume == 2)[0])
        one = len(np.where(seg_volume == 1)[0])
        return one+four, four, two

    def get_meshes(self):
        t1_volume = self.produce_data(0)
        t1_volume[np.where(t1_volume != 0)] = 1
        t1_volume = scipy.ndimage.gaussian_filter(t1_volume.astype(np.float32), sigma=2)
        t1_volume = zero_padding(t1_volume)
        verts, faces = measure.marching_cubes_classic(t1_volume, 0.4, gradient_direction='ascent')
        whole_brain = trimesh.Trimesh(verts-1, faces)
        if not whole_brain.is_watertight:
            trimesh.repair.fill_holes(whole_brain)
        whole_brain = clearup_mesh(whole_brain)

        seg_volume = self.produce_data(4)
        seg_volume[np.where(seg_volume == 2)] = 0  # peritumoral edema
        seg_volume[np.where(seg_volume == 4)] = 1  # enhancing tumor
        seg_volume = scipy.ndimage.gaussian_filter(seg_volume.astype(np.float32), sigma=2)
        seg_volume = zero_padding(seg_volume)
        verts, faces = measure.marching_cubes_classic(seg_volume, 0.6, gradient_direction='ascent')
        tumer_core = trimesh.Trimesh(verts-1, faces)
        #trimesh.repair.fill_holes(tumer_core)
        tumer_core = clearup_mesh(tumer_core)

        seg_volume = self.produce_data(4)
        seg_volume[np.where(seg_volume == 2)] = 1  # peritumoral edema
        seg_volume[np.where(seg_volume == 4)] = 1  # enhancing tumor
        seg_volume = scipy.ndimage.gaussian_filter(seg_volume.astype(np.float32), sigma=1)
        seg_volume = zero_padding(seg_volume)
        verts, faces = measure.marching_cubes_classic(seg_volume, 0.5, gradient_direction='ascent')
        whole_tumer = trimesh.Trimesh(verts-1, faces)
        trimesh.repair.fill_holes(whole_tumer)


        '''verts = np.concatenate([whole_brain.vertices, tumer_core.vertices], axis=0)
        verts_label = np.concatenate([np.zeros(shape=whole_brain.vertices.shape[0]), np.ones(shape=tumer_core.vertices.shape[0])])
        faces = np.zeros_like(tumer_core.faces)
        faces[:,0] = tumer_core.faces[:,0]
        faces[:,1] = tumer_core.faces[:,2]
        faces[:,2] = tumer_core.faces[:,1]
        faces += whole_brain.vertices.shape[0]
        faces = np.concatenate([whole_brain.faces, faces], axis=0)
        brain_without_tumer = trimesh.Trimesh(verts, faces)'''

        self.meshes = [tumer_core, whole_tumer, whole_brain]

        print(tumer_core.is_watertight)
        print(whole_tumer.is_watertight)
        print(whole_brain.is_watertight)
        return tumer_core, whole_tumer, whole_brain

    def get_tetr_mesh(self):
        self.tetr_mesh = Tetr_Mesh(self.meshes[0], self.meshes[1], self.meshes[2])
        return self.tetr_mesh
    def get_fs_t1file(self):
        return FREESURFER_SBJ_DIR + 'BraTS_' + str(self.data_id).zfill(3) + '/mri/T1.mgz'

    def get_affine(self):
        import os
        from util_package.util import load_affine_transform
        supposed_FS_reg_dir = FREESURFER_SBJ_DIR + 'BraTS_' + str(self.data_id).zfill(3) + '/mri/transforms/'
        supposed_FS_reg_file = supposed_FS_reg_dir + 'talairach.lta'
        if os.path.isfile(supposed_FS_reg_file):
            affine_talairach = load_affine_transform(supposed_FS_reg_file, start_from=9)
            freesurfer_orig = nib.load(FREESURFER_SBJ_DIR + 'BraTS_' + str(self.data_id).zfill(3) + '/mri/orig.mgz' )
            affine_orig = freesurfer_orig.affine
            return affine_talairach.dot(np.linalg.inv(affine_orig)).dot(self.affine)
        else:
            cmd = ['-subjid', 'BraTS_' + str(self.data_id).zfill(3),
                   '-i', self.data_path + self.list_of_files[0],
                   '-autorecon1', '-gcareg']
            cmd = ['-s', 'BraTS_' + str(self.data_id).zfill(3), '-canorm -careg -rmneck -skull-lta -calabel']
            command = 'recon-all'
            for i in cmd:
                command = command+' '+i
            return command+'\n'