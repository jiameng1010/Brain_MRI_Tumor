from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
from mindboggle.Subject import Subject
from Sim_Data.Sim_Data import Sim_Data
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
from util_package import util, plot, constants
from mayavi import mlab
import random
from Working_Environment.environment_variables import *

mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()
f = open('training_data.txt', 'r')
training_data_index = f.readlines()
f.close()

#for i in [335, 339, 343, 347, 351, 355, 359, 363, 367]:
for i in [331, 333, 335, 337, 339, 341, 343, 345, 347, 349]:
    print(i)
    ###
    #output_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/debug_output_dir'
    index = training_data_index[i]
    output_dir = index[:-1]
    print(output_dir)
    generator = Training_Data_Generator(None, None, None, output_dir, None)
    #generator.interpolate_displacement()
    #generator.produce_seed()
    #generator.run_BrainSim()
    #continue
    #generator.interpolate_displacement()
    output_volume = generator.undeform('/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/predicted_disp/' + str(i).zfill(5) + '.mha', scale=0.2)
    #output_volume = generator.undeform(generator.output_dir + '/SimTumor_def_inverse.mha')
    diff_volume = generator.displacement_residual('/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/predicted_disp/' + str(i).zfill(5) + '.mha')
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
