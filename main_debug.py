from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
from mindboggle.Subject import Subject
from Sim_Data.Sim_Data import Sim_Data
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
from util_package import util, plot, constants
from mayavi import mlab
import random

BraTS_dataset_dir = "/home/mjia/Researches/Volume_Segmentation/TumorMRI/MICCAI_BraTS2020_TrainingData"

Mindboggle_dataset_dir = "/home/mjia/Researches/Volume_Segmentation/mindboggle"
mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()

BrainSim_inputdata_dir = "/home/mjia/Researches/Volume_Segmentation/NITRC-multi-file-downloads/InputData"

training_data_output_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/my_training_data'

for i in range(36, 41):
    print(i)
    ###
    #output_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/debug_output_dir'
    output_dir = training_data_output_dir + '/' + str(i).zfill(5)
    generator = Training_Data_Generator(None, None, None, output_dir)
    #generator.produce_seed()
    #generator.run_BrainSim()
    #continue
    generator.interpolate_displacement()
    continue
    ####################################################################################################################
    mindboggle_whole_brain = subject_mindboggle.get_iso_surface(4)
    tumer_core, whole_tumer, whole_brain = test_data.get_meshes()
    mesh = BrainSim_inputdata.get_iso_surface(6)

    #mindboggle_affine = subject_mindboggle.get_affine_()
    mindboggle_whole_brain_v = util.affine(constants.Mindboggle_Affine_MNI152, mindboggle_whole_brain.vertices)

    mesh_vertices = util.affine(constants.BrainSim_Affine, mesh.vertices)

    BraTS_affine = test_data.get_affine()
    whole_brain_vertices = util.affine(constants.BraTS_Affine, whole_brain.vertices)
    #whole_brain_vertices = whole_brain.vertices
    #_ = test_data.get_tetr_mesh()
    #test_data.tetr_volume.debug_solve(plot=True)
    #undeformed_image = test_data.produce_undeformed(0, save_to='/home/mjia/Researches/Volume_Segmentation/TumorMRI/recon_test')
    mlab.triangular_mesh(whole_brain_vertices[:, 0], whole_brain_vertices[:, 1], whole_brain_vertices[:, 2], whole_brain.faces, color=(1, 0, 0), opacity=0.2)
    mlab.triangular_mesh(mindboggle_whole_brain_v[:, 0], mindboggle_whole_brain_v[:, 1], mindboggle_whole_brain_v[:, 2], mindboggle_whole_brain.faces, color=(0, 1, 0), opacity=0.2)
    mlab.triangular_mesh(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], mesh.faces, color=(0, 0, 1), opacity=0.2)
    mlab.show()
