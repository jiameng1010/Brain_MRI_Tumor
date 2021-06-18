import os
import subprocess
from medpy.io import load, save
import numpy as np
import nibabel as nib
from util_package import util, plot, constants
from Working_Environment.environment_variables import *

class Training_Data_Generator():
    def __init__(self, BraTS_data, Mindboggle_data, BrainSim, output_dir, source):
        if BrainSim == None:
            if source != None:
                self.output_dir = output_dir
                self.from_source(source)
            else:
                self.output_dir = output_dir
                self.from_dir()
        else:
            self.output_dir = output_dir
            self.BraTS_data = BraTS_data
            self.Mindboggle_data = Mindboggle_data
            self.BrainSim = BrainSim

            if not os.path.isdir(self.output_dir):
                subprocess.run(['mkdir', self.output_dir])
                textfile = open(self.output_dir + '/Source_info', "w")
                textfile.write('This is synthesised from:\n')
                textfile.write('TumorSim_inputdata ' + str(source[0])+'\n')
                textfile.write('Mindboggle ' + str(source[1]) + '\n')
                textfile.write('BraTS ' + str(source[2]) + '\n')
                textfile.close()

    def from_dir(self):
        source = util.parse_source_info(self.output_dir + '/Source_info')
        self.from_source(source)

    def from_source(self, source):
        from BraTS_Data.BraTS_Data import BraTS_Data
        from mindboggle.Mindboggle import Mindboggle
        from mindboggle.Subject import Subject
        from Sim_Data.Sim_Data import Sim_Data
        mb_data = Mindboggle(Mindboggle_dataset_dir)
        subject_list = mb_data.get_subject_list()
        self.BrainSim = Sim_Data(BrainSim_inputdata_dir, source[0])
        self.Mindboggle_data = Subject(mb_data, subject_list[source[1]])
        self.BraTS_data = BraTS_Data(BraTS_dataset_dir, source[2])

    def produce_seed(self):
        import scipy
        BraTS_label_volume = self.BraTS_data.produce_data(4)
        BraTS_label_volume[np.where(BraTS_label_volume == 2)] = 0  # peritumoral edema
        BraTS_label_volume[np.where(BraTS_label_volume == 4)] = 1  # enhancing tumor
        BraTS_label_volume = scipy.ndimage.gaussian_filter(BraTS_label_volume.astype(np.float32), sigma=1)
        shape = BraTS_label_volume.shape

        seed, header = load('./Training_Data_Generator/Seed.nrrd')
        seed_out = np.zeros_like(seed)
        index = np.where(seed_out == 0)
        voxels = np.concatenate([np.expand_dims(index[0], axis=1),
                                 np.expand_dims(index[1], axis=1),
                                 np.expand_dims(index[2], axis=1)], axis=1)
        transformation = np.linalg.inv(self.BraTS_data.get_affine()).dot(self.BrainSim.get_affine())
        transformed_voxels = util.affine(transformation, voxels)
        voxels_moved = transformed_voxels.astype(int)
        voxels_moved = np.concatenate([np.expand_dims(np.clip(voxels_moved[:, 0], 0.0, shape[0]), axis=1),
                                       np.expand_dims(np.clip(voxels_moved[:, 1], 0.0, shape[1]), axis=1),
                                       np.expand_dims(np.clip(voxels_moved[:, 2], 0.0, shape[2]), axis=1)], axis=1)
        x = np.linspace(0, shape[0], shape[0])
        y = np.linspace(0, shape[1], shape[1])
        z = np.linspace(0, shape[2], shape[2])
        fn = scipy.interpolate.RegularGridInterpolator((x, y, z), BraTS_label_volume)
        value = fn(voxels_moved)
        seed_out[index] = value
        seed_out = 1 * (seed_out > 0.5)
        seed_out = 1 * scipy.ndimage.morphology.binary_closing(seed_out)
        seed_out = 1 * scipy.ndimage.binary_erosion(seed_out)
        save(seed_out, self.output_dir+'/Seed.nrrd', hdr=header)

    def run_BrainSim(self):
        f = open("./Training_Data_Generator/TumorSim.xml", "r")
        xml_text = f.readlines()
        textfile = open(self.output_dir+'/TumorSim.xml', "w")
        for i in range(len(xml_text)):
            if i == 25:
                textfile.write('<input-directory>' + self.BrainSim.data_path + '</input-directory>' + '\n')
            elif i == 36:
                textfile.write('<deformation-seed>' + self.output_dir +'/Seed.nrrd' + '</deformation-seed>' + '\n')
            elif i == 31:
                textfile.write('<output-directory>' + self.output_dir + '</output-directory>' + '\n')
            else:
                textfile.write(xml_text[i] + "\n")
        textfile.close()
        subprocess.run([TumorSim_executable, self.output_dir+'/TumorSim.xml'])

    def interpolate_displacement(self):
        displacement_field1, _ = load(self.output_dir + '/SimTumor_def2.mha')
        displacement_field2, _ = load(self.output_dir + '/SimTumor_def1.mha')
        tumor_labels, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
        displacement_field = displacement_field1 + displacement_field2
        transformation = np.linalg.inv(constants.BrainSim_Affine).dot(constants.Mindboggle_Affine_MNI152)
        transformation = np.linalg.inv(self.BrainSim.get_affine()).dot(self.Mindboggle_data.get_affine())
        transformed_volume, label_manual_aseg, label_manual = util.interpolate_with_displacement(self.Mindboggle_data.get_volume_data(4).get_fdata(),
                                                                self.Mindboggle_data.get_volume_data(0).get_fdata(),
                                                                self.Mindboggle_data.get_volume_data(2).get_fdata(),
                                                                displacement_field, tumor_labels,
                                                                transformation)
        nib.save(self.Mindboggle_data.get_volume_data(4),
                 self.output_dir + '/original_t1.nii.gz')
        nib.save(self.Mindboggle_data.get_volume_data(0),
                 self.output_dir + '/original_labels_manual_aseg.nii.gz')
        nib.save(self.Mindboggle_data.get_volume_data(2),
                 self.output_dir + '/original_label_manual.nii.gz')
        nib.save(nib.Nifti1Image(transformed_volume, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_t1.nii.gz')
        nib.save(nib.Nifti1Image(label_manual_aseg, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_labels_manual_aseg.nii.gz')
        nib.save(nib.Nifti1Image(label_manual, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_label_manual.nii.gz')

        SimTumor_t1, _ = load(self.output_dir + '/SimTumor_T1.mha')
        transformed_SimTumor = util.interpolate_without_displacement(SimTumor_t1, transformation, self.Mindboggle_data.get_volume_data(4).shape)
        transformed_RealTumor = util.interpolate_without_displacement(self.BraTS_data.produce_data(0),
                                                                     np.linalg.inv(self.BraTS_data.get_affine()).dot(self.Mindboggle_data.get_affine()),
                                                                     self.Mindboggle_data.get_volume_data(4).shape)
        nib.save(nib.Nifti1Image(transformed_RealTumor, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_RealTumor.nii.gz')
        nib.save(nib.Nifti1Image(transformed_SimTumor, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_SimTumor.nii.gz')


    def interpolate(self):
        import scipy
        tumor_labels, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
        shape = tumor_labels.shape
        x = np.linspace(0, shape[0], shape[0])
        y = np.linspace(0, shape[1], shape[1])
        z = np.linspace(0, shape[2], shape[2])
        fn = scipy.interpolate.RegularGridInterpolator((x, y, z), BraTS_label_volume)

        transformation = np.linalg.inv(constants.BrainSim_Affine).dot(constants.Mindboggle_Affine_MNI152)
        value = fn(voxels_moved)