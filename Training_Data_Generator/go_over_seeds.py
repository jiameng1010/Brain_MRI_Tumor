import h5py as h5
from medpy.io import load, save
import os
import numpy as np
import copy

'''from skimage.morphology import skeletonize, thin, erosion
def thin_and_erosion(seed):
    thinned = thin(seed)
    eroded = erosion(thinned)
    return eroded'''

def half_seed(seed):
    from scipy.ndimage.morphology import binary_erosion
    org_size = np.sum(seed == 1)
    if org_size < 7000:
        return seed
    else:
        output_seed = copy.copy(seed)
        size = np.sum(output_seed == 1)
        while (size > org_size/2):
            output_seed = 1 * binary_erosion(output_seed)
            size = np.sum(output_seed == 1)
    return output_seed

#seed, _ = load('/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/my_training_data/00000/Seed.nrrd')
#print(np.sum(seed == 1))
#seed_h = half_seed(seed)


h5f = h5.File('seed_size.h5', 'r')
seed_size = h5f['seed_size'][:]
import matplotlib.pyplot as plt
plt.hist(np.asarray(seed_size), density=True, bins=30)
plt.show()


training_data_output_dir = '/media/mjia/Researches2/Brain_Tumor/my_training_data_seed_size'
seed_size = []
for entry in os.scandir(training_data_output_dir):
    output_dir = entry.path
    seed, _ = load(output_dir + '/Seed.nrrd')
    seed_size.append(np.sum(seed == 1))

h5f = h5.File('seed_size.h5', 'w')
h5f.create_dataset('seed_size', data=np.asarray(seed_size))
h5f.close()