from medpy.io import load
import numpy as np
import trimesh
from mindboggle.util import load_vtk_file

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

    def get_volume_data(self, file_index=0):
        volume, image_header = load(self.data_path + '/' + file_list[file_index])
        return volume

    def get_iso_surface(self, file_index=0):
        from skimage import measure
        volume, image_header = load(self.data_path + '/' + file_list[file_index])
        volume[np.where(volume!=0)] = 1
        t1_volume = zero_padding(volume)
        verts, faces = measure.marching_cubes_classic(t1_volume, 0.4, gradient_direction='ascent')
        return trimesh.Trimesh(verts-1, faces)

    def get_mesh(self):
        load_vtk_file(self.data_path + '/' + file_list[7])

    #def get_affine(self):
