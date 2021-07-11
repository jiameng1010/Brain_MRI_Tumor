import os
import subprocess
from medpy.io import load, save
import numpy as np
import nibabel as nib
import trimesh
from util_package import util, plot, constants
from Working_Environment.environment_variables import *

def volume_to_mesh(waped_label_volume):
    import scipy, skimage
    from util_package.util import zero_padding
    waped_label_volume = 1 * scipy.ndimage.morphology.binary_closing(waped_label_volume)
    waped_label_volume = scipy.ndimage.gaussian_filter(waped_label_volume.astype(np.float32), sigma=2)
    waped_label_volume = zero_padding(waped_label_volume)
    verts, faces = skimage.measure.marching_cubes_classic(waped_label_volume, 0.5, gradient_direction='ascent')
    mesh = trimesh.Trimesh(verts - 1, faces)
    meshes = trimesh.graph.split(mesh)
    return meshes[0]

def get_affin2unisphere(bounding_sphere):
    affine = np.zeros(shape=(4, 4))
    size = bounding_sphere.bounds[1, :] - bounding_sphere.bounds[0, :]
    scale = 1/size

    return affine

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
            elif i == 52:
                textfile.write('<deformation-initial-pressure>' + str(2.25+np.random.normal(scale=0.2)) + '</deformation-initial-pressure>' + '\n')
            elif i == 54:
                textfile.write('<deformation-damping>' + str(0.95+np.random.uniform(low=-0.01, high=0.01)) + '</deformation-damping>' + '\n')
            else:
                textfile.write(xml_text[i] + "\n")
        textfile.close()
        subprocess.run([TumorSim_executable, self.output_dir+'/TumorSim.xml'])

    def handle_displacement(self):
        try:
            load(self.output_dir + '/SimTumor_def.mha')
            load(self.output_dir + '/SimTumor_def_inverse.mha')
        except:
            displacement_1, _ = load(self.output_dir + '/SimTumor_def1.mha')
            displacement_2, _ = load(self.output_dir + '/SimTumor_def2.mha')
            displacement_inv_1, _ = load(self.output_dir + '/SimTumor_def1_inverse.mha')
            displacement_inv_2, _ = load(self.output_dir + '/SimTumor_def2_inverse.mha')

            displacement = util.cascade_diaplacements(displacement_2, displacement_1)
            displacement_inv = util.cascade_diaplacements(displacement_inv_1, displacement_inv_2)

            save(displacement, self.output_dir + '/SimTumor_def.mha')
            save(displacement_inv, self.output_dir + '/SimTumor_def_inverse.mha')

            #debugging
            '''label_volume, _ = load('/home/mjia/Researches/Volume_Segmentation/NITRC-multi-file-downloads/InputData/TumorSimInput5/labels.mha')
            label_volume1, _ = load(self.output_dir + '/SimTumor_warped_labels1.mha')
            label_volume2, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')

            from scipy.interpolate import RegularGridInterpolator
            shape = displacement_1.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), label_volume, method='nearest')

            index = np.where(label_volume >= 0)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_moved = voxels_interpolate + np.reshape(displacement, (-1, 3))
            interpolated_label = np.zeros_like(label_volume)
            voxels_moved, index = util.clap_voxels_out(voxels_moved, index, shape)
            interpolated_label[index] = fn(voxels_moved)
            diff = 1 * ((interpolated_label - label_volume2)!= 0)
            diff[np.where(label_volume2 == 5)[0]] = 0
            save(diff, self.output_dir + 'diff.mha')
            print('debug')'''





    def interpolate_displacement(self):
        displacement_field1, _ = load(self.output_dir + '/SimTumor_def1.mha')
        displacement_field2, _ = load(self.output_dir + '/SimTumor_def2.mha')
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

    def get_meshes(self):
        if not os.path.isfile(self.output_dir + '/Tumor_mesh.off'):
            waped_label_volume, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
            waped_label_volume[np.where(waped_label_volume != 5)] = 0
            waped_label_volume[np.where(waped_label_volume == 5)] = 1
            tumor_mesh = volume_to_mesh(waped_label_volume)
            tumor_mesh.export(self.output_dir + '/Tumor_mesh.off')
        else:
            tumor_mesh = trimesh.load_mesh(self.output_dir + '/Tumor_mesh.off')

        if not os.path.isfile(self.output_dir + '/Brain_mesh.off'):
            waped_label_volume, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
            waped_label_volume[np.where(waped_label_volume != 0)] = 1
            brain_mesh = volume_to_mesh(waped_label_volume)
            brain_mesh.export(self.output_dir + '/Brain_mesh.off')
        else:
            brain_mesh = trimesh.load_mesh(self.output_dir + '/Brain_mesh.off')

        return tumor_mesh, brain_mesh

    def generate_training_points(self, num_tumor_points, num_brain_points, num_querry):
        self.handle_displacement()
        tumor_mesh, brain_mesh = self.get_meshes()
        brain_pc, _ = trimesh.sample.sample_surface(brain_mesh, num_brain_points)

        PC_points, PC_face_index = trimesh.sample.sample_surface(tumor_mesh, num_tumor_points)
        PC_surface_normal = tumor_mesh.face_normals[PC_face_index, :]
        displacement_field, _ = load(self.output_dir + '/SimTumor_def.mha')
        shape = displacement_field.shape
        x = np.linspace(0, shape[0], shape[0])
        y = np.linspace(0, shape[1], shape[1])
        z = np.linspace(0, shape[2], shape[2])
        from scipy.interpolate import RegularGridInterpolator
        fn = RegularGridInterpolator((x, y, z), displacement_field)
        displacements = fn(PC_points)

        PC_points_transformed = util.affine(self.BrainSim.get_affine(), PC_points)
        displacements_transformed = util.affine(self.BrainSim.get_affine(), (PC_points + displacements)) - PC_points_transformed
        tumor_PC = [PC_points_transformed, displacements_transformed, PC_surface_normal]

        #affine2unisphere = get_affin2unisphere(brain_mesh.bounding_sphere)
        print(brain_mesh.bounds[1,:] - brain_mesh.bounds[0, :])
        querry_points1, _ = trimesh.sample.sample_surface(tumor_mesh, int(num_querry/2))
        querry_points1 = querry_points1 + 0.15*np.random.normal(size=querry_points1.shape) * (tumor_mesh.bounds[1]-tumor_mesh.bounds[0])
        querry_points1 = util.clap_voxels_out1(querry_points1, shape)
        querry_points2 = trimesh.sample.volume_mesh(brain_mesh, int(num_querry/2))
        querry_points3 = np.random.uniform(size=[num_querry-querry_points1.shape[0]-querry_points2.shape[1], 3])
        querry_points3 = util.affine(constants.Affine_uniform2mindboggle, querry_points3)
        querry_points = np.concatenate([querry_points1, querry_points2, querry_points3], axis=0)
        querry_points_displacements = fn(querry_points)
        querry_points_transformed = util.affine(self.BrainSim.get_affine(), querry_points)
        querry_points_displacements_transformed = util.affine(self.BrainSim.get_affine(), (querry_points + querry_points_displacements)) - querry_points_transformed
        querry_PC = [querry_points_transformed, querry_points_displacements_transformed]

        self.save_training_pointclouds(tumor_PC, brain_pc, querry_PC)
        self.write_augmented_info(tumor_mesh.volume, trimesh.Trimesh(tumor_mesh.vertices + fn(tumor_mesh.vertices), tumor_mesh.faces).volume)
        return tumor_PC, brain_pc, querry_PC

    def write_augmented_info(self, tumor_volume, tissue_volume):
        source = util.parse_source_info(self.output_dir + '/Source_info')
        textfile = open(self.output_dir + '/augmented_info', "w")
        textfile.write('This is synthesised from:\n')
        textfile.write('TumorSim_inputdata ' + str(source[0]) + '\n')
        textfile.write('Mindboggle ' + str(source[1]) + '\n')
        textfile.write('BraTS ' + str(source[2]) + '\n')
        textfile.write('tumor_volume ' + str(tumor_volume) + '\n')
        textfile.write('tissue_volume ' + str(tissue_volume) + '\n')
        textfile.close()

    def save_training_pointclouds(self, tumor_PC, brain_pc, querry_PC):
        import h5py
        hf = h5py.File(self.output_dir+'/training_pointclouds.h5', 'w')
        hf.create_dataset('tumor_PC', data=tumor_PC[0])
        hf.create_dataset('tumor_PC_displacement', data=tumor_PC[1])
        hf.create_dataset('tumor_PC_surfacenormal', data=tumor_PC[2])
        hf.create_dataset('brain_PC', data=brain_pc)
        hf.create_dataset('querry_PC', data=querry_PC[0])
        hf.create_dataset('querry_PC_displacement', data=querry_PC[1])
        hf.close()

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