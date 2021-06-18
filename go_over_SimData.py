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

#f = open('command_SimData.sh', 'w')
for i in range(5):
    BrainSim_inputdata = Sim_Data(BrainSim_inputdata_dir, i+1)

    affine = BrainSim_inputdata.get_volume_data()
    #print(affine)
    #f.write(affine)

#f.close()