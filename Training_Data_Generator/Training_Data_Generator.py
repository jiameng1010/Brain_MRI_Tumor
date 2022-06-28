import os
import subprocess
from medpy.io import load, save
import numpy as np
import nibabel as nib
import trimesh
from util_package import util, plot, constants
from Working_Environment.environment_variables import *

def volume_to_mesh(waped_label_volume, threshold=0.5):
    import scipy, skimage
    from util_package.util import zero_padding
    waped_label_volume = 1 * scipy.ndimage.morphology.binary_closing(waped_label_volume)
    waped_label_volume = 1 * scipy.ndimage.binary_fill_holes(waped_label_volume)
    #waped_label_volume = scipy.ndimage.gaussian_filter(waped_label_volume.astype(np.float32), sigma=2)
    waped_label_volume = zero_padding(waped_label_volume)
    verts, faces = skimage.measure.marching_cubes_classic(waped_label_volume, threshold, gradient_direction='ascent')
    mesh = trimesh.Trimesh(verts - 1, faces)
    #meshes = trimesh.graph.split(mesh)
    return mesh

def volume_to_mesh_continus(waped_label_volume, threshold=0.5):
    import scipy, skimage
    from util_package.util import zero_padding
    waped_label_volume = zero_padding(waped_label_volume)
    verts, faces = skimage.measure.marching_cubes_classic(waped_label_volume, threshold, gradient_direction='ascent')
    mesh = trimesh.Trimesh(verts - 1, faces)
    #meshes = trimesh.graph.split(mesh)
    return mesh

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
        def half_seed(seed):
            from scipy.ndimage.morphology import binary_erosion
            import copy
            org_size = np.sum(seed == 1)
            if org_size < 1000:
                return seed
            else:
                output_seed = copy.copy(seed)
                size = np.sum(output_seed == 1)
                while (size > org_size / 2):
                    output_seed = 1 * binary_erosion(output_seed)
                    size = np.sum(output_seed == 1)
            return output_seed
        import scipy
        BraTS_label_volume = self.BraTS_data.produce_data(4)
        BraTS_label_volume[np.where(BraTS_label_volume == 2)] = 0  # peritumoral edema
        BraTS_label_volume[np.where(BraTS_label_volume == 4)] = 1  # enhancing tumor
        #BraTS_label_volume = scipy.ndimage.gaussian_filter(BraTS_label_volume.astype(np.float32), sigma=1)
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
        x = np.linspace(0, shape[0]-1, shape[0])
        y = np.linspace(0, shape[1]-1, shape[1])
        z = np.linspace(0, shape[2]-1, shape[2])
        fn = scipy.interpolate.RegularGridInterpolator((x, y, z), BraTS_label_volume, method='nearest')
        voxels_moved, index = util.clap_voxels_out(voxels_moved, index, shape)
        value = fn(voxels_moved)
        seed_out[index] = value
        print(np.sum(seed_out))####################
        #seed_out = 1 * (seed_out > 0.5)
        seed_out = 1 * scipy.ndimage.binary_fill_holes(seed_out)
        seed_out = 1 * scipy.ndimage.morphology.binary_opening(seed_out)
        print(np.sum(seed_out))  ####################
        seed_size = np.sum(seed_out == 1)
        if seed_size == 0:
            return 1
        print(np.sum(seed_out))  ####################
        seed_out, num_features = scipy.ndimage.measurements.label(seed_out)
        max_connected = 0
        for i in range(1, num_features+1):
            if len(np.where(seed_out == i)[0]) > max_connected:
                max_connected = len(np.where(seed_out == i)[0])
                max_index = i
        print(len(np.where(seed_out == max_index)[0]) / len(np.where(seed_out != 0)[0]))
        seed_out_single_connected = np.zeros_like(seed_out)
        seed_out_single_connected[np.where(seed_out == max_index)] = 1
        seed_out_single_connected = half_seed(seed_out_single_connected)
        print(np.sum(seed_out_single_connected))
        save(seed_out_single_connected, self.output_dir+'/Seed.nrrd', hdr=header)
        return 0

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
                textfile.write('<deformation-initial-pressure>' + str(2.0+np.random.uniform(low=-0.4, high=0.4)) + '</deformation-initial-pressure>' + '\n')
            elif i == 54:
                textfile.write('<deformation-damping>' + str(0.90) + '</deformation-damping>' + '\n')
            elif i == 68:
                textfile.write('<white-matter-tensor-multiplier>' + str(10.0+np.random.uniform(low=-2.0, high=2.0)) + '</white-matter-tensor-multiplier>' + '\n')
            elif i == 69:
                textfile.write('<gray-matter-tensor-multiplier>' + str(1.0+np.random.uniform(low=-0.2, high=0.2)) + '</gray-matter-tensor-multiplier>' + '\n')
            else:
                textfile.write(xml_text[i] + "\n")
        textfile.close()
        subprocess.run([TumorSim_executable, self.output_dir+'/TumorSim.xml'])

    def displacement_to_svf(self):
        #if self.is_svf_good():
        #    return
        import scipy
        displacement_field, _ = load(self.output_dir + '/SimTumor_def.mha')
        displacement_field = np.concatenate([np.expand_dims(scipy.ndimage.gaussian_filter(displacement_field[:,:,:,0], sigma=3), axis=-1),
                                            np.expand_dims(scipy.ndimage.gaussian_filter(displacement_field[:,:,:,1], sigma=3), axis=-1),
                                             np.expand_dims(scipy.ndimage.gaussian_filter(displacement_field[:,:,:,2], sigma=3), axis=-1)], axis=-1)
        #displacement_field = np.flip(displacement_field, axis=0)
        #displacement_field = np.flip(displacement_field, axis=1)
        #displacement_field = np.flip(displacement_field, axis=2)
        #img = nib.Nifti1Image(displacement_field, np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        #nib.save(img, self.output_dir + '/SimTumor_def_smooth.nii.gz')
        save(displacement_field, self.output_dir + '/SimTumor_def_smooth.mha')
        '''subprocess.run([MIRTK_EXECUTABLE,
                        'convert-dof',
                        self.output_dir + '/SimTumor_def_smooth.nii.gz',
                        self.output_dir + '/displacement.nii.gz',
                        '-input-format', 'disp_voxel',
                        '-output-format', 'mirtk'])
                        #'-ss', '-smooth',
                        #'-steps', '512',])
                        #'-terms', '8',
                        #'-iters', '16'])'''
        subprocess.run([MIRTK_EXECUTABLE,
                        'calculate-logarithmic-map',
                        self.output_dir + '/SimTumor_def_smooth.mha',
                        self.output_dir + '/output_svf.mha',
                        '-terms', '2',
                        '-iters', '1',
                        '-steps', '512',
                        '-threads', '4',
                        '-smooth'
                        ])
        svf, _ = load(self.output_dir + '/output_svf.mha')
        #svf = np.concatenate([np.expand_dims(scipy.ndimage.gaussian_filter(svf[:,:,:,0], sigma=10), axis=-1),
        #                                    np.expand_dims(scipy.ndimage.gaussian_filter(svf[:,:,:,1], sigma=10), axis=-1),
        #                                     np.expand_dims(scipy.ndimage.gaussian_filter(svf[:,:,:,2], sigma=10), axis=-1)], axis=-1)
        #save(svf, self.output_dir + '/output_svf.mha')
        #subprocess.run([MIRTK_EXECUTABLE,
        #                'calculate-exponential-map',
        #                self.output_dir + '/output_svf.mha',
        #                self.output_dir + '/output_disp.mha',
        #                '-steps', '512'])
        #self.is_svf_good()

    def is_svf_good(self):
        if not os.path.isfile(self.output_dir + '/output_svf.mha'):
            return False
        displacement_field, _ = load(self.output_dir + '/output_svf.mha')
        range = np.max(displacement_field) - np.min(displacement_field)
        print(range)
        if range<1e-5 or range>300:
            return False
        else:
            return True

    def handle_displacement(self):
        #displacement_1, _ = load(self.output_dir + '/SimTumor_def1.mha')
        #displacement_out = util.displacement_to_inv(displacement_1)
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
            #displacement = displacement_1
            #displacement_inv = displacement_inv_1

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

            # debug
            '''diff = np.zeros_like(displacement_inv)
            label_volume, _ = load(
                '/home/mjia/Researches/Volume_Segmentation/NITRC-multi-file-downloads/InputData/TumorSimInput5/labels.mha')
            index = np.where(label_volume >= 0)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_moved = voxels_interpolate + np.reshape(displacement_inv, (-1, 3))
            shape = displacement_1.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            from scipy.interpolate import RegularGridInterpolator
            fn = RegularGridInterpolator((x, y, z), displacement_inv, method='nearest')
            voxels_moved, index = util.clap_voxels_out(voxels_moved, index, shape)
            value = fn(voxels_moved)
            diff[index] = value - np.reshape(displacement_inv[index], (-1, 3))
            save(diff, self.output_dir + 'diff.mha')
            print('debug')'''

            # debugging
            '''label_volume, _ = load('/home/mjia/Researches/Volume_Segmentation/NITRC-multi-file-downloads/InputData/TumorSimInput5/labels.mha')
            label_volume1, _ = load(self.output_dir + '/SimTumor_warped_labels1.mha')
            label_volume2, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')

            from scipy.interpolate import RegularGridInterpolator
            shape = displacement_1.shape
            x = np.linspace(0, shape[0] - 1, shape[0])
            y = np.linspace(0, shape[1] - 1, shape[1])
            z = np.linspace(0, shape[2] - 1, shape[2])
            fn = RegularGridInterpolator((x, y, z), label_volume2, method='nearest')

            index = np.where(label_volume >= 0)
            voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                                 np.expand_dims(index[1], axis=1),
                                                 np.expand_dims(index[2], axis=1)], axis=1)
            voxels_moved = voxels_interpolate + np.reshape(displacement_inv, (-1, 3))
            interpolated_label = np.zeros_like(label_volume)
            voxels_moved, index = util.clap_voxels_out(voxels_moved, index, shape)
            interpolated_label[index] = fn(voxels_moved)
            diff = 1 * ((interpolated_label - label_volume)!= 0)
            diff[np.where(label_volume == 5)[0]] = 0
            save(diff, self.output_dir + 'diff.mha')
            print('debug')'''


    def undeform(self, deformation_file, scale=1.0):
        displacement_field, _ = load(deformation_file)
        input_volume, _ = load(self.output_dir + '/transformed_t1.nii.gz')
        input_volume = scale * input_volume
        displacement_transformed = util.interpolate_with_affine(displacement_field,
                                                                np.linalg.inv(self.BrainSim.get_affine()).dot(self.Mindboggle_data.get_affine()),
                                                                [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2], 3])
        output_volume = util.interpolate_volume_displacement(input_volume, displacement_transformed)
        nib.save(nib.Nifti1Image(output_volume, self.Mindboggle_data.affine),
                 deformation_file.split('.')[0] + 'undeformed.nii.gz')
        nib.save(nib.Nifti1Image(input_volume, self.Mindboggle_data.affine),
                 deformation_file.split('.')[0] + 'deformed.nii.gz')
        return output_volume

    def interpolate_displacement(self, with_skull=False):
        self.handle_displacement()
        displacement_field, _ = load(self.output_dir + '/SimTumor_def.mha')
        tumor_labels, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
        transformation = np.linalg.inv(self.BrainSim.get_affine()).dot(self.Mindboggle_data.get_affine())
        if with_skull:
            which_file = 7
        else:
            which_file = 5
        transformed_volume, label_manual_aseg, label_manual = util.interpolate_with_displacement(self.Mindboggle_data.get_volume_data(5).get_fdata(),
                                                                self.Mindboggle_data.get_volume_data(-1).get_fdata(),
                                                                self.Mindboggle_data.get_volume_data(3).get_fdata(),
                                                                displacement_field, tumor_labels,
                                                                transformation)
        transformed_volume_with_skull, label_manual_aseg, label_manual = util.interpolate_with_displacement(self.Mindboggle_data.get_volume_data(7).get_fdata(),
                                                                self.Mindboggle_data.get_volume_data(-1).get_fdata(),
                                                                self.Mindboggle_data.get_volume_data(3).get_fdata(),
                                                                displacement_field, tumor_labels,
                                                                transformation)
        nib.save(self.Mindboggle_data.get_volume_data(with_skull),
                 self.output_dir + '/original_t1.nii.gz')
        nib.save(self.Mindboggle_data.get_volume_data(1),
                 self.output_dir + '/original_labels_manual_aseg.nii.gz')
        nib.save(self.Mindboggle_data.get_volume_data(3),
                 self.output_dir + '/original_label_manual.nii.gz')
        nib.save(nib.Nifti1Image(transformed_volume, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_t1.nii.gz')
        nib.save(nib.Nifti1Image(transformed_volume_with_skull, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_t1_with_skull.nii.gz')
        nib.save(nib.Nifti1Image(label_manual_aseg, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_labels_manual_aseg.nii.gz')
        nib.save(nib.Nifti1Image(label_manual, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_label_manual.nii.gz')

        SimTumor_t1, _ = load(self.output_dir + '/SimTumor_T1.mha')
        transformed_SimTumor = util.interpolate_without_displacement(SimTumor_t1, transformation, self.Mindboggle_data.get_volume_data(7).shape)
        nib.save(nib.Nifti1Image(transformed_SimTumor, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_SimTumor.nii.gz')
        transformed_RealTumor = util.interpolate_without_displacement(self.BraTS_data.produce_data(0),
                                                                     np.linalg.inv(self.BraTS_data.get_affine()).dot(self.Mindboggle_data.get_affine()),
                                                                     self.Mindboggle_data.get_volume_data(7).shape)
        nib.save(nib.Nifti1Image(transformed_RealTumor, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_RealTumor.nii.gz')

        transformed_RealTumor = util.interpolate_without_displacement(self.BraTS_data.produce_data(4),
                                                                     np.linalg.inv(self.BraTS_data.get_affine()).dot(self.Mindboggle_data.get_affine()),
                                                                     self.Mindboggle_data.get_volume_data(7).shape, method='nearest')
        nib.save(nib.Nifti1Image(transformed_RealTumor, self.Mindboggle_data.affine),
                 self.output_dir + '/transformed_RealTumor_label.nii.gz')

    def get_meshes(self, get_edma_mesh=False):
        if True:#not os.path.isfile(self.output_dir + '/Tumor_mesh.off'):
            waped_label_volume, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
            waped_label_volume[np.where(waped_label_volume != 5)] = 0
            waped_label_volume[np.where(waped_label_volume == 5)] = 1
            tumor_mesh = volume_to_mesh(waped_label_volume)
            tumor_mesh.export(self.output_dir + '/Tumor_mesh.off')
        else:
            tumor_mesh = trimesh.load_mesh(self.output_dir + '/Tumor_mesh.off')

        if True:#not os.path.isfile(self.output_dir + '/Brain_mesh.off'):
            waped_label_volume, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
            waped_label_volume[np.where(waped_label_volume != 0)] = 1
            brain_mesh = volume_to_mesh(waped_label_volume)
            brain_mesh.export(self.output_dir + '/Brain_mesh.off')
        else:
            brain_mesh = trimesh.load_mesh(self.output_dir + '/Brain_mesh.off')

        if True: # not os.path.isfile(self.output_dir + '/Edma_mesh.off'):
            edma_prob, _ = load(self.output_dir + '/SimTumor_prob4.mha')
            core_prob, _ = load(self.output_dir + '/SimTumor_prob5.mha')
            waped_label_volume = (edma_prob + core_prob) / 65536
            edma_mesh = volume_to_mesh_continus(waped_label_volume, threshold=0.5)
            edma_mesh.export(self.output_dir + '/Edma_mesh.off')
        else:
            edma_mesh = trimesh.load_mesh(self.output_dir + '/Edma_mesh.off')

        if get_edma_mesh:
            return tumor_mesh, brain_mesh, edma_mesh
        else:
            return tumor_mesh, brain_mesh

    def generate_training_points(self, num_tumor_points, num_brain_points, num_querry):
        #if os.path.isfile(self.output_dir+'/training_pointclouds.h5'):
        #    return
        self.handle_displacement()
        tumor_mesh, brain_mesh, edma_mesh = self.get_meshes(get_edma_mesh=True)
        brain_pc, _ = trimesh.sample.sample_surface(brain_mesh, num_brain_points)
        brain_pc = util.affine(self.BrainSim.get_affine(), brain_pc)
        edma_pc, _ = trimesh.sample.sample_surface(edma_mesh, num_tumor_points)
        edma_pc = util.affine(self.BrainSim.get_affine(), edma_pc)

        PC_points, PC_face_index = trimesh.sample.sample_surface(tumor_mesh, num_tumor_points)
        PC_surface_normal = tumor_mesh.face_normals[PC_face_index, :]
        displacement_field, _ = load(self.output_dir + '/output_svf.mha')
        shape = displacement_field.shape
        x = np.linspace(0, shape[0]-1, shape[0])
        y = np.linspace(0, shape[1]-1, shape[1])
        z = np.linspace(0, shape[2]-1, shape[2])
        from scipy.interpolate import RegularGridInterpolator
        fn = RegularGridInterpolator((x, y, z), displacement_field)
        displacements = fn(PC_points)

        PC_points_transformed = util.affine(self.BrainSim.get_affine(), PC_points)
        displacements_transformed = util.affine(self.BrainSim.get_affine(), (PC_points + displacements)) - PC_points_transformed
        tumor_PC = [PC_points_transformed, displacements_transformed, PC_surface_normal]

        #affine2unisphere = get_affin2unisphere(brain_mesh.bounding_sphere)
        print(brain_mesh.bounds[1,:] - brain_mesh.bounds[0, :])
        querry_points1, _ = trimesh.sample.sample_surface(tumor_mesh, int(num_querry/2))
        querry_points1 = querry_points1 + 0.3*np.random.normal(size=querry_points1.shape) * (tumor_mesh.bounds[1]-tumor_mesh.bounds[0])
        querry_points1 = util.clap_voxels_out1(querry_points1, shape)
        querry_points2 = trimesh.sample.volume_mesh(brain_mesh, int(num_querry/3))
        querry_points3 = np.random.uniform(size=[num_querry-querry_points1.shape[0]-querry_points2.shape[0], 3])
        querry_points3 = util.affine(constants.Affine_uniform2mindboggle, querry_points3)
        querry_points = np.concatenate([querry_points1, querry_points2, querry_points3], axis=0)
        querry_points_displacements = fn(querry_points)
        querry_points_transformed = util.affine(self.BrainSim.get_affine(), querry_points)
        querry_points_displacements_transformed = util.affine(self.BrainSim.get_affine(), (querry_points + querry_points_displacements)) - querry_points_transformed
        querry_PC = [querry_points_transformed, querry_points_displacements_transformed]

        self.save_training_pointclouds(tumor_PC, brain_pc, edma_pc, querry_PC)
        self.write_augmented_info(tumor_mesh.volume, trimesh.Trimesh(tumor_mesh.vertices + fn(tumor_mesh.vertices), tumor_mesh.faces).volume)
        print(trimesh.Trimesh(tumor_mesh.vertices + fn(tumor_mesh.vertices), tumor_mesh.faces).volume / tumor_mesh.volume)
        return tumor_PC, brain_pc, edma_pc, querry_PC

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

    def save_training_pointclouds(self, tumor_PC, brain_pc, edma_pc, querry_PC):
        #if os.path.isfile(self.output_dir + '/training_pointclouds_0.3.h5'):
        #    subprocess.run(['rm', self.output_dir + '/training_pointclouds_0.3.h5'])
        import h5py
        saving_file_name = self.output_dir+'/training_pointclouds_svf_0.3.h5'

        if os.path.isfile(saving_file_name):
            subprocess.run(['rm', saving_file_name])
        hf = h5py.File(saving_file_name, 'w')
        hf.create_dataset('tumor_PC', data=tumor_PC[0])
        hf.create_dataset('tumor_PC_displacement', data=tumor_PC[1])
        hf.create_dataset('tumor_PC_surfacenormal', data=tumor_PC[2])
        hf.create_dataset('brain_PC', data=brain_pc)
        hf.create_dataset('edma_PC', data=edma_pc)
        hf.create_dataset('querry_PC', data=querry_PC[0])
        hf.create_dataset('querry_PC_displacement', data=querry_PC[1])
        hf.close()

    def interpolate(self):
        import scipy
        tumor_labels, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
        shape = tumor_labels.shape
        x = np.linspace(0, shape[0]-1, shape[0])
        y = np.linspace(0, shape[1]-1, shape[1])
        z = np.linspace(0, shape[2]-1, shape[2])
        fn = scipy.interpolate.RegularGridInterpolator((x, y, z), BraTS_label_volume)

        transformation = np.linalg.inv(constants.BrainSim_Affine).dot(constants.Mindboggle_Affine_MNI152)
        value = fn(voxels_moved)


    def displacement_residual(self, deformation_file):
        displacement_inv, _ = load(deformation_file)
        displacement_field, _ = load(self.output_dir + '/SimTumor_def.mha')
        out_volume = np.zeros_like(displacement_inv)
        from scipy.interpolate import RegularGridInterpolator
        index = np.where(out_volume[:,:,:,0] == 0)
        voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                             np.expand_dims(index[1], axis=1),
                                             np.expand_dims(index[2], axis=1)], axis=1)
        voxel_moved = voxels_interpolate + np.reshape(displacement_field, [-1, 3])

        shape = displacement_inv.shape
        x = np.linspace(0, shape[0] - 1, shape[0])
        y = np.linspace(0, shape[1] - 1, shape[1])
        z = np.linspace(0, shape[2] - 1, shape[2])
        fn = RegularGridInterpolator((x, y, z), displacement_inv)
        voxel_moved, index = util.clap_voxels_out(voxel_moved, index, shape)
        displacement_value = fn(voxel_moved)
        out_volume[index] = displacement_value
        out_volume = displacement_field + out_volume

        save(out_volume, deformation_file.split('.')[0] + 'difference.mha')
        return out_volume

    def get_tumor_size(self):
        seed_size = self.get_seed_size()
        waped_label_volume, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
        voxel_tumor_size = len(np.where(waped_label_volume == 5)[0])
        #waped_label_volume, _ = load(self.output_dir + '/Seed.nrrd')
        #voxel_tumor_size = len(np.where(waped_label_volume == 1)[0])
        return seed_size, voxel_tumor_size

    def get_tumor_size(self):
        waped_label_volume, _ = load(self.output_dir + '/SimTumor_warped_labels2.mha')
        voxel_tumor_size = len(np.where(waped_label_volume == 5)[0])
        #waped_label_volume, _ = load(self.output_dir + '/Seed.nrrd')
        #voxel_tumor_size = len(np.where(waped_label_volume == 1)[0])
        seed, _ = load(self.output_dir + '/Seed.nrrd')
        seed_size = np.sum(seed)
        return seed_size, voxel_tumor_size

    def sythesis_t1(self):
        factor1, _ = load(self.output_dir + '/SimTumor_prob1.mha')
        factor2, _ = load(self.output_dir + '/SimTumor_prob2.mha')
        factor3, _ = load(self.output_dir + '/SimTumor_prob3.mha')
        factor4, _ = load(self.output_dir + '/SimTumor_prob4.mha')
        factor5, _ = load(self.output_dir + '/SimTumor_prob5.mha')
        factor6, _ = load(self.output_dir + '/SimTumor_prob6.mha')

        image1, _ = load(self.BrainSim.data_path + '/textures/t1_1.mha')
        image2, _ = load(self.BrainSim.data_path + '/textures/t1_2.mha')
        image3, _ = load(self.BrainSim.data_path + '/textures/t1_3.mha')
        image4, _ = load(self.BrainSim.data_path + '/textures/t1_4.mha')
        image5, _ = load(self.BrainSim.data_path + '/textures/t1_5.mha')
        image6, _ = load(self.BrainSim.data_path + '/textures/t1_bg.mha')
        real_tumor_image = util.interpolate_without_displacement(self.BraTS_data.produce_data(0),
                                                                     np.linalg.inv(self.BraTS_data.get_affine()).dot(self.BrainSim.get_affine()),
                                                                     image6.shape)

        subprocess.run(['mkdir', self.output_dir+'/tmp'])
        tmp_img = nib.load(self.BrainSim.get_fs_t1file())
        nib.save(tmp_img, self.output_dir+'/tmp/TumorSim.nii.gz')
        tmp_img = nib.load(self.Mindboggle_data.get_fs_t1file())
        nib.save(tmp_img, self.output_dir+'/tmp/Mindboggle.nii.gz')
        tmp_img = nib.load(self.BraTS_data.get_fs_t1file())
        nib.save(tmp_img, self.output_dir+'/tmp/BraTS.nii.gz')
        subprocess.run([MIRTK_EXECUTABLE, 'register',
                        self.output_dir+'/tmp/TumorSim.nii.gz',
                        self.output_dir+'/tmp/Mindboggle.nii.gz',
                        '-dofout', self.output_dir + '/tmp/deformation_m2t',
                        '-output', self.output_dir + '/tmp/registered_Mindboggle_t1.nii.gz',
                        '-be', '0.001'])
        subprocess.run([MIRTK_EXECUTABLE, 'register',
                        self.output_dir + '/tmp/TumorSim.nii.gz',
                        self.output_dir + '/tmp/BraTS.nii.gz',
                        '-dofout', self.output_dir + '/tmp/deformation_B2t',
                        '-output', self.output_dir + '/tmp/registered_BraTS_t1.nii.gz',
                        '-be', '0.001'])

        sythesised_t1 = (factor1/65536) * image1 + \
            (factor2/65536) * image2 + \
            (factor3/65536) * image3 + \
            (factor4/65536) * image4 + \
            (factor5/65536) * real_tumor_image / 12# + \
            #image6 / 8
        save(sythesised_t1, self.output_dir+'/sythesised_t1.mha')

    def sythesis_t1_mindboggle(self):
        subprocess.run(['mkdir', self.output_dir + '/tmp'])

        Mt1 = nib.load(self.output_dir + '/transformed_t1.nii.gz')
        Mt1_img = Mt1.get_fdata()
        Mt1_label = nib.load(self.output_dir + '/transformed_labels_manual_aseg.nii.gz').get_fdata()
        Mt1_img[np.where(Mt1_label==-1)] = 1024
        nib.save(nib.Nifti1Image(Mt1_img, Mt1.affine), self.output_dir+'/tmp/transformed_t1.nii.gz')

        Bt1 = nib.load(self.output_dir + '/transformed_RealTumor.nii.gz')
        Bt1_img = Bt1.get_fdata()
        Bt1_label = nib.load(self.output_dir + '/transformed_RealTumor_label.nii.gz').get_fdata()
        Bt1_label[np.where(Bt1_label==3)] = 1
        Bt1_label[np.where(Bt1_label==4)] = 1
        Bt1_img[np.where(Bt1_label==1)] = 512
        nib.save(nib.Nifti1Image(Bt1_img, Bt1.affine), self.output_dir+'/tmp/transformed_RealTumor.nii.gz')

        subprocess.run([MIRTK_EXECUTABLE, 'register',
                        self.output_dir+'/tmp/transformed_t1.nii.gz',
                        self.output_dir+'/tmp/transformed_RealTumor.nii.gz',
                        '-dofout', self.output_dir + '/tmp/deformation_RealTumor2t1',
                        '-output', self.output_dir + '/tmp/registered_transformed_RealTumor.nii.gz',
                        '-be', '0.0001', '-tp', '0.00001'])
        tmp_img = nib.load(self.output_dir + '/transformed_RealTumor.nii.gz')
        nib.save(tmp_img, self.output_dir+'/tmp/transformed_RealTumor.nii.gz')
        subprocess.run([MIRTK_EXECUTABLE, 'transform-image',
                        self.output_dir + '/transformed_RealTumor.nii.gz',
                        self.output_dir + '/tmp/registered_transformed_RealTumor.nii.gz',
                        '-dofin', self.output_dir + '/tmp/deformation_RealTumor2t1',
                        '-interp', 'Linear'])

        subprocess.run([MIRTK_EXECUTABLE, 'transform-image',
                        self.output_dir + '/transformed_RealTumor_label.nii.gz',
                        self.output_dir + '/tmp/registered_transformed_RealTumor_label.nii.gz',
                        '-dofin', self.output_dir + '/tmp/deformation_RealTumor2t1',
                        '-interp', 'NN'])

        self.interpolate_displacement(with_skull=True)
        Mt1_file = nib.load(self.output_dir + '/transformed_t1_with_skull.nii.gz')
        Mt1 = Mt1_file.get_fdata()
        mask1 = np.zeros_like(Mt1)
        mask1[np.where(Mt1_label==-1)] = 0.5
        B_label = np.squeeze(nib.load(self.output_dir + '/tmp/registered_transformed_RealTumor_label.nii.gz').get_fdata())
        mask2 = np.zeros_like(Mt1)
        mask2[np.where(B_label>=1)] = 0.5
        mask = mask1 + mask2
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=2)
        Bt1 = np.squeeze(nib.load(self.output_dir + '/tmp/registered_transformed_RealTumor.nii.gz').get_fdata())

        gray_scale_rate = 0.7 * np.mean(Mt1[np.where(mask1 + mask2 ==1)]) / np.mean(Bt1[np.where(mask1 + mask2 ==1)])
        sythesised_t1 = (np.ones_like(mask) - mask) * Mt1 + gray_scale_rate * mask * Bt1
        nib.save(nib.Nifti1Image(sythesised_t1, Mt1_file.affine), self.output_dir + '/sythesised_t1.nii.gz')

        Mt1_file = nib.load(self.output_dir + '/transformed_t1.nii.gz')
        Mt1 = Mt1_file.get_fdata()
        sythesised_t1 = (np.ones_like(mask) - mask) * Mt1 + gray_scale_rate * mask * Bt1
        nib.save(nib.Nifti1Image(sythesised_t1, Mt1_file.affine), self.output_dir + '/sythesised_t1_without_skull.nii.gz')


    def delete_file(self, filename):
        if os.path.isfile(self.output_dir+'/'+filename):
            subprocess.run(['rm', self.output_dir+'/'+filename])

    def get_seed_size(self):
        seed, header = load(self.output_dir + '/Seed.nrrd')
        return np.sum(seed == 1)

    def make_eval_ready(self, eval_dir):
        if not os.path.isdir(eval_dir):
            subprocess.run(['mkdir', eval_dir])
        subprocess.run(['cp', self.Mindboggle_data.get_volume_data_file(-1), eval_dir])
        subprocess.run(['cp', self.Mindboggle_data.get_volume_data_file(7), eval_dir])
        subprocess.run(['cp', self.output_dir + '/transformed_labels_manual_aseg.nii.gz', eval_dir])
        subprocess.run(['cp', self.output_dir + '/SimTumor_prob4.mha', eval_dir])
        subprocess.run(['cp', self.output_dir + '/SimTumor_prob5.mha', eval_dir])
        subprocess.run(['cp', self.output_dir + '/SimTumor_warped_labels2.mha', eval_dir])
        subprocess.run(['cp', self.output_dir + '/Source_info', eval_dir])
        subprocess.run(['cp', self.output_dir + '/sythesised_t1.nii.gz', eval_dir + '/sythesised_t1_0001.nii.gz'])
        subprocess.run(['cp', self.output_dir + '/sythesised_t1_without_skull.nii.gz', eval_dir + '/sythesised_t1_without_skull.nii.gz'])
        subprocess.run(['cp', self.output_dir + '/SimTumor_def.mha',
                        eval_dir + '/SimTumor_def.mha'])
        subprocess.run(['cp', self.output_dir + '/SimTumor_def_inverse.mha',
                        eval_dir + '/SimTumor_def_inverse.mha'])


