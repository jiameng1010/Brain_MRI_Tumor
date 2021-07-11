from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
from mindboggle.Subject import Subject
from Sim_Data.Sim_Data import Sim_Data
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
from util_package import util, plot, constants
from mayavi import mlab
import random
import os
from Working_Environment.environment_variables import *

mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()

for i in range(0, 100):
#for entry in os.scandir(training_data_output_dir):
    '''BrainSim_inputdata_index = random.randint(1, 5)
    BrainSim_inputdata = Sim_Data(BrainSim_inputdata_dir, BrainSim_inputdata_index)

    subject_index = random.randint(0,100)#, 78)
    print(subject_index)
    subject_mindboggle = Subject(mb_data, subject_list[subject_index])

    BraTS_index = random.randint(1, 369)
    test_data = BraTS_Data(BraTS_dataset_dir, BraTS_index)'''
    #_ = test_data.get_meshes()
    #_ = test_data.get_tetr_mesh()
    #test_data.tetr_mesh.debug_solve(plot=True)
    #undeformed_image = test_data.produce_undeformed(0, save_to='/home/mjia/Researches/Volume_Segmentation/TumorMRI/recon_test')

    ####################################################################################################################
    output_dir = training_data_output_dir + '/' + str(i).zfill(5)
    #output_dir = entry.path

    print(output_dir)
    if not os.path.isfile(output_dir + '/transformed_t1.nii.gz'):
        continue

    #generator = Training_Data_Generator(test_data, subject_mindboggle, BrainSim_inputdata, output_dir,
    #                                    source=[BrainSim_inputdata_index, subject_index, BraTS_index])
    generator = Training_Data_Generator(None, None, None, output_dir, None)
    test_data = generator.BraTS_data
    subject_mindboggle = generator.Mindboggle_data
    BrainSim_inputdata = generator.BrainSim

    try:
        tumor_mesh, brain_mesh = generator.get_meshes()
        tumor_PC, brain_pc, querry_PC = generator.generate_training_points(2000, 2000, 2000)
        continue
    except:
        continue
    '''try:
        #generator.produce_seed()
        #generator.run_BrainSim()
        import os
        if not os.path.isfile(output_dir + '/SimTumor_T1.mha'):
            continue
        #generator.interpolate_displacement()
        tumor_mesh, brain_mesh = generator.get_meshes()
        tumor_PCs = generator.generate_training_points()
        continue
    except:
        continue'''
    ####################################################################################################################
    mindboggle_whole_brain = subject_mindboggle.get_iso_surface(4)
    tumer_core, whole_tumer, whole_brain = test_data.get_meshes()
    mesh = BrainSim_inputdata.get_iso_surface(6)
    #affine = subject_mindboggle.affine.dot()

    affine = subject_mindboggle.get_affine()
    mindboggle_whole_brain_v = util.affine(affine, mindboggle_whole_brain.vertices)

    affine = BrainSim_inputdata.get_affine()
    mesh_vertices = util.affine(affine, mesh.vertices)

    BraTS_affine = test_data.get_affine()
    whole_brain_vertices = util.affine(BraTS_affine, whole_brain.vertices)
    #whole_brain_vertices = whole_brain.vertices
    #_ = test_data.get_tetr_mesh()
    #test_data.tetr_volume.debug_solve(plot=True)
    #undeformed_image = test_data.produce_undeformed(0, save_to='/home/mjia/Researches/Volume_Segmentation/TumorMRI/recon_test')
    mlab.triangular_mesh(whole_brain_vertices[:, 0], whole_brain_vertices[:, 1], whole_brain_vertices[:, 2], whole_brain.faces, color=(1, 0, 0), opacity=0.2)
    mlab.triangular_mesh(mindboggle_whole_brain_v[:, 0], mindboggle_whole_brain_v[:, 1], mindboggle_whole_brain_v[:, 2], mindboggle_whole_brain.faces, color=(0, 1, 0), opacity=0.2)
    mlab.triangular_mesh(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], mesh.faces, color=(0, 0, 1), opacity=0.2)

    tumor_mesh_vertices = util.affine(affine, tumor_mesh.vertices)
    mlab.triangular_mesh(tumor_mesh_vertices[:, 0], tumor_mesh_vertices[:, 1], tumor_mesh_vertices[:, 2], tumor_mesh.faces, color=(0, 1, 1),
                         opacity=0.2)
    mlab.points3d(querry_PC[0][:, 0], querry_PC[0][:, 1], querry_PC[0][:, 2], scale_factor=2)
    mlab.quiver3d(querry_PC[0][:, 0], querry_PC[0][:, 1], querry_PC[0][:, 2],
                  querry_PC[1][:, 0], querry_PC[1][:, 1], querry_PC[1][:, 2], scale_factor=1)
    mlab.show()
