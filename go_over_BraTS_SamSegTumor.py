from BraTS_Data.BraTS_Data import BraTS_Data
from mindboggle.Mindboggle import Mindboggle
import sys, os, subprocess
from Working_Environment.environment_variables import *
sys.path.append(SAMSEG_DIR)
from freesurfer.samseg import Samseg, ProbabilisticAtlas, SamsegTumor

mb_data = Mindboggle(Mindboggle_dataset_dir)
subject_list = mb_data.get_subject_list()

'''f = open('command_BraTS.sh', 'w')
for i in range(369):
    test_data = BraTS_Data(BraTS_dataset_dir, i+1)
    affine = test_data.get_affine()
    print(affine)
    f.write(affine)
f.close()'''

def SamsegTumor_BraTS(test_data):
    output_folder = test_data.data_path + '/SamsegTumor_noskull_output'
    if os.path.isdir(output_folder):
        subprocess.run(['rm', '-r', output_folder])
    subprocess.run(['mkdir', output_folder])

    ss = SamsegTumor.SamsegTumor(imageFileNames=[test_data.get_filename(0),
                                                 test_data.get_filename(1),
                                                 test_data.get_filename(2),
                                                 test_data.get_filename(3)],
                                 atlasDir='/tmp/atlas_tumor_noskull',
                                 savePath=output_folder,
                                 modeNames=['t1', 't1ce', 't2', 'flair'],
                                 useMeanConstr=False, useShapeModel=False, useMRF=True,
                                 savePosteriors=True)
    ss.segment()

#f = open('command_BraTS.sh', 'w')
'''tumor_size_all = []
subprocess.run(['cp', '-r', SAMSEG_ATLAS_DIR1, '/tmp'])
subprocess.run(['mv', '/tmp/20Subjects_smoothing2_down2_smoothingForAffine2', '/tmp/atlas'])
SamsegTumor.makeTumorAtlas('/tmp/atlas',
                           '/tmp/atlas_tumor',
                           skull=False)'''
import numpy as np
from Working_Environment.environment_variables import *
sys.path.append(SAMSEG_DIR)
from os.path import join
from freesurfer.samseg import Samseg, ProbabilisticAtlas, SamsegTumor
from freesurfer.samseg import gems
from shutil import copyfile
def makeTumorAtlas(atlasDirOrig,atlasDirNew,tumorPrior=0.2,skull=True):
    os.makedirs(atlasDirNew,exist_ok=True)
    levelFiles = [join(atlasDirOrig, i) for i in os.listdir(atlasDirOrig) if "level" in i.lower()]
    for meshFile in levelFiles:
        meshFileOut = meshFile.replace(atlasDirOrig, atlasDirNew)
        meshcoll = gems.KvlMeshCollection()
        meshcoll.read(meshFile)
        mesh = meshcoll.reference_mesh
        import copy
        mesh_alphas = copy.copy(mesh.alphas)
        label_to_delete = [1, 2, 8, 23]
        #mesh_alphas[:, 0] = mesh_alphas[:, 0] + mesh_alphas[:, 1]
        #mesh_alphas[:, 1] = np.zeros_like(mesh_alphas[:, 1])
        for label in label_to_delete:
            #mesh_alphas /= np.expand_dims(np.ones_like(mesh_alphas[:, label]) - mesh_alphas[:, label], axis=1)
            mesh_alphas[:,0] += mesh_alphas[:,label]
            mesh_alphas[:,label] = np.zeros_like(mesh_alphas[:,label])
        label_to_delete = [23, 8, 2, 1]
        for label in label_to_delete:
            mesh_alphas = np.delete(mesh_alphas, label, axis=1)
        newAlphas = np.zeros((mesh_alphas.shape[0], mesh_alphas.shape[1] + 1))
        newAlphas[:, :-1] = mesh_alphas.copy()
        if skull:
            tumorPossible = mesh_alphas[:,:3].sum(axis=-1)<0.5
        else:
            tumorPossible = mesh_alphas[:, 0] < 0.5  # can only have tumor inside the brain
        newAlphas[tumorPossible, -1] = tumorPrior
        newAlphas[tumorPossible, :-1] *= (1-tumorPrior)
        sumAlphas = newAlphas[tumorPossible, :].sum(axis=1)
        newAlphas[tumorPossible, :] /= (sumAlphas[:, np.newaxis])
        mesh.alphas = newAlphas
        meshcoll.reference_mesh.alphas = mesh.alphas
        meshcoll.write(meshFileOut.replace(".gz", ""))

    sharedParamFile = join(atlasDirOrig, "sharedGMMParameters.txt")
    sharedParamLines = open(sharedParamFile).readlines()[:-1]
    sharedParamLines.append("Tumor 3 Tumor\n")
    with open(join(atlasDirNew, "sharedGMMParameters.txt"), "w") as f:
        for line in sharedParamLines:
            f.write(line)
    LUTFile = join(atlasDirOrig, "compressionLookupTable.txt")
    LUTLines = open(LUTFile).readlines()[:-4]
    #LUTLines.append("99 %d Tumor                        255    0    0  255\n" % len(LUTLines))
    with open(join(atlasDirNew, "compressionLookupTable.txt"), "w") as f:
        for line in LUTLines:
            index_org = int(line.split(' ')[1])
            if index_org == 0:
                f.write(line)
            else:
                if index_org <= 8:
                    index = index_org - 2
                elif index_org <= 23:
                    index = index_org - 3
                else:
                    index = index_org - 4
                line_new = line.split(' ')
                line_new[1] = str(index)
                line_out = ''
                for item in line_new:
                    line_out += item+' '
                f.write(line_out)
        f.write("99 40 Tumor                        255    0    0  255\n")
    otherFiles = [join(atlasDirOrig, i) for i in os.listdir(atlasDirOrig) if
                  not "level" in i.lower() and not "parameters.txt" in i.lower() and not "compressionlookuptable.txt" in i.lower()]
    _ = [copyfile(i, i.replace(atlasDirOrig, atlasDirNew)) for i in otherFiles]

#subprocess.run(['rm', '-r', '/tmp/atlas_tumor_noskull'])
makeTumorAtlas('/tmp/20Subjects_smoothing2_down2_smoothingForAffine2', '/tmp/atlas_noskull')
for i in range(369):
    test_data = BraTS_Data(BraTS_dataset_dir, i+1)
    if not os.path.isfile(test_data.data_path + '/SamsegTumor_output/seg.mgz'):
        continue
    SamsegTumor_BraTS(test_data)

