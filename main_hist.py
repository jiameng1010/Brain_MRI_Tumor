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

#for i in range(0, 100):
training_data_output_dir = '/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/my_training_data'
tumor_size_all = []
seed_size_all = []
rate_all = []
for entry in os.scandir(training_data_output_dir):
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
    #output_dir = training_data_output_dir + '/' + str(i).zfill(5)
    output_dir = entry.path

    print(output_dir)
    if not os.path.isfile(output_dir + '/SimTumor_def.mha'):
        continue

    #generator = Training_Data_Generator(test_data, subject_mindboggle, BrainSim_inputdata, output_dir,
    #                                    source=[BrainSim_inputdata_index, subject_index, BraTS_index])
    generator = Training_Data_Generator(None, None, None, output_dir, None)
    test_data = generator.BraTS_data
    subject_mindboggle = generator.Mindboggle_data
    BrainSim_inputdata = generator.BrainSim

    #if not os.path.isfile(generator.output_dir + '/Seed_fill.nrrd'):
    #    continue
    seed_size, voxel_tumor_size = generator.get_tumor_size()
    #seed_size = generator.get_seed_size()
    tumor_size_all.append(voxel_tumor_size)
    seed_size_all.append(seed_size)
    rate_all.append(voxel_tumor_size/seed_size)

import matplotlib.pyplot as plt
import numpy as np
plt.hist(np.asarray(tumor_size_all), density=True, bins=30)
plt.show()


