from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
from mindboggle.Subject import Subject
from Sim_Data.Sim_Data import Sim_Data
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator
from util_package import util, plot, constants
from mayavi import mlab
import random
import os, subprocess
from Working_Environment.environment_variables import *

mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()

#for i in range(0, 100):
training_data_output_dir = '/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/my_training_data_seed_size'
#training_data_output_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/training_data'
view_files_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/training_data/files_viewing'
eval_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7'
num_of_error = 0
for i in range(80):
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
    #if not os.path.isfile(output_dir + '/augmented_info'):
    #    continue

    #generator = Training_Data_Generator(test_data, subject_mindboggle, BrainSim_inputdata, output_dir,
    #                                    source=[BrainSim_inputdata_index, subject_index, BraTS_index])
    #generator = Training_Data_Generator(None, None, None, output_dir, None)
    source = util.parse_source_info(output_dir + '/Source_info')
    generator = Training_Data_Generator(None, None, None, output_dir, [source[0], 79+i%20, source[2]])

    #subprocess.run(['cp', generator.output_dir+'/sythesised_t1.nii.gz', view_files_dir])
    #subprocess.run(['mv', view_files_dir+'/sythesised_t1.nii.gz', view_files_dir+'/'+str(i)+'sythesised_t1.nii.gz'])
    #continue
    test_data = generator.BraTS_data
    subject_mindboggle = generator.Mindboggle_data
    BrainSim_inputdata = generator.BrainSim

    #generator.displacement_to_svf()
    #generator.delete_file('output_svf.nii.gz')
    '''try:
        #error = generator.produce_seed()
        #num_of_error += error
        #continue
        #generator.handle_displacement()
        #generator.interpolate_displacement()
        #generator.displacement_to_svf()
        #tumor_mesh, brain_mesh = generator.get_meshes()
        tumor_PC, brain_pc, querry_PC = generator.generate_training_points(2000, 2000, 2000)
        #continue
    except:
        continue'''
    print(subject_mindboggle.id)

    tumor_mesh, brain_mesh = generator.get_meshes()
    generator.interpolate_displacement()
    generator.sythesis_t1_mindboggle()
    generator.make_eval_ready(eval_dir + '/' + output_dir.split('/')[-1])
    #generator.product_brain_mask()
    continue
    if not os.path.isfile(output_dir + '/SimTumor_T1.mha'):
        continue
    #generator.displacement_to_svf()
    tumor_PC, brain_pc, edma_pc, querry_PC = generator.generate_training_points(2000, 2000, 2000)
    #generator.displacement_to_svf()
    #continue
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
    mindboggle_whole_brain = subject_mindboggle.get_iso_surface(5)
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
    #mlab.triangular_mesh(whole_brain_vertices[:, 0], whole_brain_vertices[:, 1], whole_brain_vertices[:, 2], whole_brain.faces, color=(0, 1, 0), opacity=0.2)
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.triangular_mesh(mindboggle_whole_brain_v[:, 0], mindboggle_whole_brain_v[:, 1], mindboggle_whole_brain_v[:, 2], mindboggle_whole_brain.faces, color=(0, 0, 1), opacity=0.1)
    #mlab.triangular_mesh(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], mesh.faces, color=(0, 0, 1), opacity=0.2)



    tumor_mesh_vertices = util.affine(affine, tumor_mesh.vertices)
    #mlab.triangular_mesh(tumor_mesh_vertices[:, 0], tumor_mesh_vertices[:, 1], tumor_mesh_vertices[:, 2], tumor_mesh.faces, color=(1, 0, 0),
    #                     opacity=0.2)
    mlab.triangular_mesh(tumor_mesh_vertices[:, 0], tumor_mesh_vertices[:, 1], tumor_mesh_vertices[:, 2], tumor_mesh.faces, color=(0.8, 0.3, 0),
                         opacity=0.2)

    #mlab.points3d(edma_pc[::2, 0], edma_pc[::2, 1], edma_pc[::2, 2], color=(0, 1, 0), scale_factor=2)
    #mlab.points3d(tumor_PC[0][::2, 0], tumor_PC[0][::2, 1], tumor_PC[0][::2, 2], color=(1, 0, 0), scale_factor=2)

    mlab.points3d(querry_PC[0][::2, 0], querry_PC[0][::2, 1], querry_PC[0][::2, 2], scale_factor=2,  color=(0.6, 0.6, 0.6))
    mlab.quiver3d(querry_PC[0][::2, 0], querry_PC[0][::2, 1], querry_PC[0][::2, 2],
                  querry_PC[1][::2, 0], querry_PC[1][::2, 1], querry_PC[1][::2, 2], scale_factor=1, mode='arrow')
    mlab.show()


training_data_output_dir = '/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/new_data/my_training_data'
'''for entry in os.scandir(training_data_output_dir):
    break
    output_dir = entry.path

    print(output_dir)
    if not os.path.isfile(output_dir + '/augmented_info'):
        continue

    #generator = Training_Data_Generator(test_data, subject_mindboggle, BrainSim_inputdata, output_dir,
    #                                    source=[BrainSim_inputdata_index, subject_index, BraTS_index])
    generator = Training_Data_Generator(None, None, None, output_dir, None)
    test_data = generator.BraTS_data
    subject_mindboggle = generator.Mindboggle_data
    BrainSim_inputdata = generator.BrainSim

    generator.interpolate_displacement()
    continue'''