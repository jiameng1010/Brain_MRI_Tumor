import sys
sys.path.append('..')
from Working_Environment.environment_variables import *
sys.path.append(SAMSEG_DIR)
from freesurfer.samseg import Samseg, ProbabilisticAtlas, SamsegTumor

import logging

log = logging.getLogger('log.txt')

#SamsegTumor.makeTumorAtlas('/home/mjia/Working_Tools/freesurfer/average/samseg/20Subjects_smoothing2_down2_smoothingForAffine2',
#                        '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/00000/samseg3/atlas_tumor')
for i in range(2, 20):
    ss = SamsegTumor.SamsegTumor(imageFileNames=['/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/'+str(i).zfill(5)+'/sythesised_t1_0001.nii.gz'],
                     atlasDir='/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/00000/samseg3/atlas_tumor',
                     savePath='/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/'+str(i).zfill(5)+'/samseg3',
                               useMeanConstr=False, useShapeModel=False)
    # imageToImageTransformMatrix=affine_v2v)
    ss.segment()