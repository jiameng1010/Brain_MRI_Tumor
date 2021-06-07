import os
import nibabel as nib
import vtk
import numpy as np

from mayavi import mlab
from mayavi.modules.surface import Surface

def load_subject_list(data_path):
    subject_text_file = os.path.join(data_path, 'subject_list_Mindboggle101.txt')
    with open(subject_text_file, 'r') as f:
        subject_list_org = f.readlines()
    subject_list = []
    for item in subject_list_org:
        if item[-2:] == '\t\n':
            subject_list.append(item[:-2].replace(' ', ''))
        else:
            subject_list.append(item[:-1].replace(' ', ''))
    return subject_list

def parse_id(subject_id):
    strs = str.split(subject_id, '-')
    index = int(strs[-1])
    group = ''
    for item in strs[:-1]:
        group = group + item + '-'
    group = group[:-1]
    return group, index

def parse_group(group_name):
    if group_name == 'MMRR-21':
        return group_name
    if group_name == 'NKI-RS-22':
        return group_name
    if group_name == 'NKI-TRT-20':
        return group_name
    if group_name == 'OASIS-TRT-20':
        return group_name
    else:
        return 'Extra-18'

def load_gz_file(filename):
    return nib.load(filename)

def load_vtk_file(filename):
    # load a vtk file as input
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    print('npoints:', output.GetNumberOfPoints())
    print('ncells:', output.GetNumberOfCells())
    print('nscalars:', reader.GetNumberOfScalarsInFile())
    print('ntensors:', reader.GetNumberOfTensorsInFile())
    print('ScalarName:', reader.GetScalarsNameInFile(0))
    print('TensorName:', reader.GetTensorsNameInFile(0))

    vertices = np.asarray(output.GetPoints().GetData())
    faces = np.reshape(np.asarray(output.GetCells().GetData()), [-1, 5])

    fig = mlab.figure()
    engine = mlab.get_engine()
    vtk_file_reader = engine.open(filename)
    surface = Surface()
    engine.add_filter(surface, vtk_file_reader)
    mlab.show()
    return 0

#load_vtk_file('/home/mjia/Downloads/mindboggle/mindboggle_manually_labeled_individual_brains/MMRR-21_surfaces/MMRR-21-3/lh.labels.DKT31.manual.vtk')