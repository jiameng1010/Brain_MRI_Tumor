from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
from mindboggle.Subject import Subject
from Sim_Data.Sim_Data import Sim_Data
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
from util_package import util, plot, constants
from mayavi import mlab
import random
import numpy as np

BraTS_dataset_dir = "/home/mjia/Researches/Volume_Segmentation/TumorMRI/MICCAI_BraTS2020_TrainingData"

Mindboggle_dataset_dir = "/home/mjia/Researches/Volume_Segmentation/mindboggle"
mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()

BrainSim_inputdata_dir = "/home/mjia/Researches/Volume_Segmentation/NITRC-multi-file-downloads/InputData"

training_data_output_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/my_training_data'

'''f = open('command.sh', 'w')
for i in range(101):
    subject_mindboggle = Subject(mb_data, subject_list[i])

    affine = subject_mindboggle.get_affine()
    print(affine)
    f.write(affine)

f.close()'''
aff = True

subject_mindboggle0 = Subject(mb_data, subject_list[50])
if aff:
    affine0 = subject_mindboggle0.get_affine()
else:
    affine0 = np.eye(4)
mindboggle_whole_brain0 = subject_mindboggle0.get_iso_surface(4)
mindboggle_whole_brain_v0 = util.affine(affine0, mindboggle_whole_brain0.vertices)

subject_mindboggle1 = Subject(mb_data, subject_list[10])
if aff:
    affine1 = subject_mindboggle1.get_affine()
else:
    affine1 = np.eye(4)
mindboggle_whole_brain1 = subject_mindboggle1.get_iso_surface(4)
mindboggle_whole_brain_v1 = util.affine(affine1, mindboggle_whole_brain1.vertices)

'''subject_mindboggle2 = Subject(mb_data, subject_list[13])
if aff:
    affine2 = subject_mindboggle2.get_affine()
else:
    affine2 = np.eye(4)
mindboggle_whole_brain2 = subject_mindboggle2.get_iso_surface(4)
mindboggle_whole_brain_v2 = util.affine(affine2, mindboggle_whole_brain2.vertices)'''

subject_0 = BraTS_Data(BraTS_dataset_dir, 5)
if aff:
    affine0 = subject_0.get_affine()
else:
    affine0 = np.eye(4)
#_, affine0 = subject_0.produce_data(0, ret_affine=True)
tumer_core, whole_tumer, whole_brain0 = subject_0.get_meshes()
mindboggle_whole_brain_v2 = util.affine(affine0, whole_brain0.vertices)
import trimesh
mindboggle_whole_brain2 = trimesh.Trimesh(mindboggle_whole_brain_v2, whole_brain0.faces)

mlab.triangular_mesh(mindboggle_whole_brain_v0[:, 0], mindboggle_whole_brain_v0[:, 1], mindboggle_whole_brain_v0[:, 2],
                     mindboggle_whole_brain0.faces, color=(1, 0, 0), opacity=0.2)
mlab.triangular_mesh(mindboggle_whole_brain_v1[:, 0], mindboggle_whole_brain_v1[:, 1], mindboggle_whole_brain_v1[:, 2],
                     mindboggle_whole_brain1.faces, color=(0, 1, 0), opacity=0.2)
mlab.triangular_mesh(mindboggle_whole_brain_v2[:, 0], mindboggle_whole_brain_v2[:, 1], mindboggle_whole_brain_v2[:, 2],
                     mindboggle_whole_brain2.faces, color=(0, 0, 1), opacity=0.2)
mlab.show()