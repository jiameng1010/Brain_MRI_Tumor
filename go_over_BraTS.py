from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
from mindboggle.Subject import Subject
from Sim_Data.Sim_Data import Sim_Data
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
from util_package import util, plot, constants
from mayavi import mlab
import random
import numpy as np
from Working_Environment.environment_variables import *

mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()

'''f = open('command_BraTS.sh', 'w')
for i in range(369):
    test_data = BraTS_Data(BraTS_dataset_dir, i+1)
    affine = test_data.get_affine()
    print(affine)
    f.write(affine)
f.close()'''

#f = open('command_BraTS.sh', 'w')
tumor_size_all = []
for i in range(369):
    test_data = BraTS_Data(BraTS_dataset_dir, i+1)
    tumor_size = test_data.get_tumor_size()
    print(tumor_size)
    tumor_size_all.append(tumor_size[0])

import matplotlib.pyplot as plt
import numpy as np
plt.hist(np.asarray(tumor_size_all), density=True, bins=30)
plt.show()

'''subject_0 = BraTS_Data(BraTS_dataset_dir, 5)
if aff:
    affine0 = subject_0.get_affine()
else:
    affine0 = np.eye(4)
_, affine0 = subject_0.produce_data(0, ret_affine=True)
tumer_core, whole_tumer, whole_brain0 = subject_0.get_meshes()
whole_brain_v0 = util.affine(affine0, whole_brain0.vertices)
whole_brain0_faces  = whole_brain0.faces'''

'''subject_1 = BraTS_Data(BraTS_dataset_dir, 250)
if aff:
    affine1 = subject_1.get_affine()
    affine1 = subject_1.produce_data(0, True)
else:
    affine1 = np.eye(4)
tumer_core, whole_tumer, whole_brain1 = subject_1.get_meshes()
whole_brain_v1 = util.affine(affine1, whole_brain1.vertices)
whole_brain1_faces = whole_brain1.faces

subject_2 = BraTS_Data(BraTS_dataset_dir, 50)
if aff:
    affine2 = subject_2.get_affine()
else:
    affine2 = np.eye(4)
tumer_core, whole_tumer, whole_brain2 = subject_2.get_meshes()
whole_brain_v2 = util.affine(affine2, whole_brain2.vertices)'''

import nibabel as nib
from skimage import measure
from util_package.util import load_affine_transform
volume = nib.load('/home/mjia/Researches/Volume_Segmentation/subjects/BraTS_050/mri/orig.mgz')
affine = volume.affine
volume = volume.get_fdata()
verts, faces = measure.marching_cubes_classic(volume, 0.4, gradient_direction='ascent')
#whole_brain_v2 = verts
#whole_brain_v0 = util.affine(affine, verts)
whole_brain_v0 = verts
affine = load_affine_transform('/home/mjia/Researches/Volume_Segmentation/subjects/BraTS_050/mri/transforms/talairach.lta', 9)
whole_brain_v0 = util.affine(affine, whole_brain_v0)
whole_brain0_faces  = faces

volume = nib.load('/home/mjia/Researches/Volume_Segmentation/subjects/BraTS_005/mri/orig.mgz')
affine = volume.affine
volume = volume.get_fdata()
verts, faces = measure.marching_cubes_classic(volume, 0.4, gradient_direction='ascent')
#whole_brain_v2 = verts
#whole_brain_v1 = util.affine(affine, verts)
whole_brain_v1 = verts
affine = load_affine_transform('/home/mjia/Researches/Volume_Segmentation/subjects/BraTS_005/mri/transforms/talairach.lta', 9)
whole_brain_v1 = util.affine(affine, whole_brain_v1)
whole_brain1_faces  = faces

volume = nib.load('/home/mjia/Researches/Volume_Segmentation/subjects/HLN-12-5/mri/orig.mgz')
affine = volume.affine
volume = volume.get_fdata()
verts, faces = measure.marching_cubes_classic(volume, 0.4, gradient_direction='ascent')
#whole_brain_v2 = verts
#whole_brain_v2 = util.affine(affine, verts)
whole_brain_v2 = verts
affine = load_affine_transform('/home/mjia/Researches/Volume_Segmentation/subjects/HLN-12-5/mri/transforms/talairach.lta', 9)
whole_brain_v2 = util.affine(affine, whole_brain_v2)
whole_brain2_faces  = faces


mlab.triangular_mesh(whole_brain_v0[:, 0], whole_brain_v0[:, 1], whole_brain_v0[:, 2],
                     whole_brain0_faces, color=(1, 0, 0), opacity=0.2)
mlab.triangular_mesh(whole_brain_v1[:, 0], whole_brain_v1[:, 1], whole_brain_v1[:, 2],
                     whole_brain1_faces, color=(0, 1, 0), opacity=0.2)
mlab.triangular_mesh(whole_brain_v2[:, 0], whole_brain_v2[:, 1], whole_brain_v2[:, 2],
                     whole_brain2_faces, color=(0, 0, 1), opacity=0.2)
mlab.show()