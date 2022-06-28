import h5py
import os
import numpy as np
from Working_Environment.environment_variables import *
from util_package import util, constants
from Training_Data_Generator.Training_Data_Generator import Training_Data_Generator

collection_dir = ''

#scale = 1.0 / np.asarray([160, 200, 179])

#for i in range(0, 100):
list_of_data_path = []
num_of_training_data = 0
tumor_PC_all = []
tumor_PC_displacement_all = []
tumor_PC_surfacenormal_all = []
brain_PC_all = []
edma_PC_all = []
querry_PC_all = []
querry_PC_displacement_all = []
affine_matrix_all = []
for entry in os.scandir(training_data_output_dir):
    if int(entry.name)<80:
        continue
    output_dir = entry.path

    training_data_file = output_dir + '/training_pointclouds_svf_0.3.h5'
    #training_data_file = training_data_file.replace(' ', '\ ')

    if os.path.isfile(training_data_file):
        generator = Training_Data_Generator(None, None, None, output_dir, None)
        affine_matrix = np.linalg.inv(constants.Affine_uniform2talairach)

        list_of_data_path.append(output_dir)
        hf = h5py.File(training_data_file, 'r')
        tumor_PC = hf['tumor_PC'][:]
        tumor_PC = util.affine(affine_matrix, tumor_PC)
        tumor_PC_displacement = hf['tumor_PC_displacement'][:]
        tumor_PC_displacement = util.rotation(affine_matrix, tumor_PC_displacement)
        tumor_PC_surfacenormal = hf['tumor_PC_surfacenormal'][:]
        tumor_PC_surfacenormal = util.rotation(affine_matrix, tumor_PC_surfacenormal)
        brain_PC = hf['brain_PC'][:]
        brain_PC = util.affine(affine_matrix, brain_PC)
        edma_PC = hf['edma_PC'][:]
        edma_PC = util.affine(affine_matrix, brain_PC)

        querry_PC = hf['querry_PC'][:]
        indices = np.random.permutation(querry_PC.shape[0])
        querry_PC = util.affine(affine_matrix, querry_PC)
        querry_PC_displacement = hf['querry_PC_displacement'][:]
        querry_PC_displacement = util.rotation(affine_matrix, querry_PC_displacement)
        querry_PC = querry_PC[indices[:2000]]
        querry_PC_displacement = querry_PC_displacement[indices[:2000]]
        # subprocess.run(['cp', training_data_file, './'+str(num_of_training_data).zfill(5)+'.h5'])
        # subprocess.run(['mv', './'+str(num_of_training_data).zfill(5)+'.h5', '/home/mjia/Researches/Volume_Segmentation/my_training_data_PC'])

        tumor_PC_all.append(np.expand_dims(tumor_PC, axis=0))
        tumor_PC_displacement_all.append(np.expand_dims(tumor_PC_displacement, axis=0))
        tumor_PC_surfacenormal_all.append(np.expand_dims(tumor_PC_surfacenormal, axis=0))
        brain_PC_all.append(np.expand_dims(brain_PC, axis=0))
        edma_PC_all.append(np.expand_dims(edma_PC, axis=0))
        querry_PC_all.append(np.expand_dims(querry_PC, axis=0))
        querry_PC_displacement_all.append(np.expand_dims(querry_PC_displacement, axis=0))
        affine_matrix_all.append(np.expand_dims(generator.BrainSim.get_affine(), axis=0))

        print(output_dir)
    num_of_training_data += 1

f = open('training_data_index_tumor_size_SVF.txt', 'w')
for item in list_of_data_path:
    f.write(item + '\n')
f.close()

tumor_PC_all = np.concatenate(tumor_PC_all, axis=0)
tumor_PC_displacement_all = np.concatenate(tumor_PC_displacement_all, axis=0)
tumor_PC_surfacenormal_all = np.concatenate(tumor_PC_surfacenormal_all, axis=0)
brain_PC_all = np.concatenate(brain_PC_all, axis=0)
edma_PC_all = np.concatenate(edma_PC_all, axis=0)
querry_PC_all = np.concatenate(querry_PC_all, axis=0)
querry_PC_displacement_all = np.concatenate(querry_PC_displacement_all, axis=0)
affine_matrix_all = np.concatenate(affine_matrix_all, axis=0)

#scaleing
disp_min = np.min(querry_PC_displacement_all)
disp_max = np.max(querry_PC_displacement_all)
max = max(np.abs(disp_min), np.abs(disp_max))
querry_PC_displacement_all = querry_PC_displacement_all / max
print(max)
#max of SVF is 1.2063787751955717
#max of DISP is 0.6088199579835727


hf = h5py.File('ALL_training_pointclouds_0.3_edma_tumor_size_SVF.h5', 'w')
#hf.create_dataset('data_dir_list', data=list_of_data_path)
hf.create_dataset('tumor_PC', data=tumor_PC_all)
hf.create_dataset('tumor_PC_displacement', data=tumor_PC_displacement_all)
hf.create_dataset('tumor_PC_surfacenormal', data=tumor_PC_surfacenormal_all)
hf.create_dataset('brain_PC', data=brain_PC_all)
hf.create_dataset('edma_PC', data=edma_PC_all)
hf.create_dataset('querry_PC', data=querry_PC_all)
hf.create_dataset('querry_PC_displacement', data=querry_PC_displacement_all)
hf.create_dataset('affine_matrix', data=affine_matrix_all)
hf.close()


