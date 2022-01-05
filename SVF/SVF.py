import os, subprocess
import numpy as np
from datetime import datetime
from medpy.io import load, save
from scipy.ndimage import geometric_transform
from Working_Environment.environment_variables import *

def inverse_displacement(displacement):
    shape = displacement.shape
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    grid = np.transpose(grid, [1, 2, 3, 0])

    def shift_function(output_coords):
        return (output_coords[0] - 50, output_coords[1] - 50, output_coords[2] - 50, output_coords[3])
    transformed_field = geometric_transform(grid, shift_func)
    return 0

def half_displacement(displacement):
    output_displacement = displacement / 2
    def shift_function(input_coords):
        shift = output_displacement[input_coords[0], input_coords[1], input_coords[2], :]
        return (input_coords[0] + shift[0], input_coords[1] + shift[1], input_coords[2] + shift[2], input_coords[3])
    shifted_displacement = geometric_transform(displacement, shift_function)
    output_displacement = displacement - shifted_displacement / 2
    return output_displacement

def p2_displacement(displacement):
    def shift_function(input_coords):
        shift = displacement[input_coords[0], input_coords[1], input_coords[2], :]
        return (input_coords[0] + shift[0], input_coords[1] + shift[1], input_coords[2] + shift[2], input_coords[3])
    shifted_displacement = geometric_transform(displacement, shift_function)
    return shifted_displacement

def half_displacement(displacement):
    return displacement - (displacement + p2_displacement(displacement))/4

# displacement: (n1, n2, n3, 3) float
def displacement_to_svf(displacement, num_steps=10):
    d = displacement
    for i in range(num_steps-1):
        d_shift = p2_displacement(d)
        d_p2 = (d + d_shift) / 2
        d = d - d_p2/(num_steps-i)
    return num_steps * d

def p_displacement(displacement, p):
    def shift_function(input_coords):
        shift = p[input_coords[0], input_coords[1], input_coords[2], :]
        return (input_coords[0] + shift[0], input_coords[1] + shift[1], input_coords[2] + shift[2], input_coords[3])
    shifted_displacement = geometric_transform(displacement, shift_function)
    return shifted_displacement

def svf_to_displacement(displacement, num_steps=10):
    d_out = np.zeros_like(displacement)
    for i in range(num_steps):
        d_out += (1/num_steps) * p_displacement(displacement, d_out)
    return d_out

def main1():
    deformation_file = 'SimTumor_def1.mha'
    subprocess.run([MIRTK_EXECUTABLE,
                    'calculate-logarithmic-map',
                    deformation_file,
                    'output_svf.nii.gz',
                    '-ss', '-jac',
                    '-steps', '1024',
                    '-smooth',
                    '-terms', '4',
                    '-iters', '16'])
    print('done1')
    #displacement_field = svf_to_displacement()

def main2():
    svf_file = 'output_svf.nii.gz'
    subprocess.run([MIRTK_EXECUTABLE,
                    'calculate-exponential-map',
                    svf_file,
                    'output_disp.nii.gz',
                    '-euler', '-steps', '64',])
    print('done2')

def transform():
    disp_file = 'output_disp.nii.gz'
    disp, _ = load(disp_file)
    save(disp, 'output_deformation.mha')

def transform_svf():
    disp_file = 'output_svf.nii.gz'
    disp, _ = load(disp_file)
    save(disp, 'output_svf.mha')

if __name__ == "__main__":
    a = datetime.now()
    main1()
    b = datetime.now()
    print(b-a)
    main2()
    c = datetime.now()
    print(c-b)
    transform()
    transform_svf()