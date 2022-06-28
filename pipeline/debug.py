from medpy.io import load, save
import nibabel as nib
import numpy as np
import scipy.ndimage

#disp, _ = load('./test_data/BraTS20_Training_001/reg_atlse/deformation_atlas.mha')
#mean0 = np.mean(disp[:,:,0])
#mean1 = np.mean(disp[:,:,1])
#mean2 = np.mean(disp[:,:,2])
#disp = disp - np.expand_dims(np.expand_dims(np.asarray([mean0, mean1, mean2]), axis=0), axis=0)
#save(disp, './test_data/BraTS20_Training_001/reg_atlse/deformation_atlas_demean.mha')

img = nib.load('/home/mjia/Researches/Volume_Segmentation/subjects/ATROPOS_TEMPLATE/mri/T1.nii.gz')
seg_in = nib.load('/home/mjia/Researches/Volume_Segmentation/mindboggle/mindboggle_atlases/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz')
seg_out = np.zeros_like(img)
affine = np.linalg.inv(seg_in.affine).dot(img.affine)
rotation = affine[:3, :3]
shift = affine[:3, 3]
from scipy.ndimage import geometric_transform
def shift_function(input_coords):
    output_coords = rotation.dot(input_coords) + shift
    return (output_coords[0], output_coords[1], output_coords[2])
shifted_volume = geometric_transform(seg_in.get_fdata(), shift_function, output_shape=[256, 256, 256])


img = nib.load('/home/mjia/Researches/Volume_Segmentation/subjects/ATROPOS_TEMPLATE/mri/aseg.auto.mgz')
nib.save(img, '/home/mjia/Researches/Volume_Segmentation/subjects/ATROPOS_TEMPLATE/mri/aseg.auto.nii.gz')

labels, _ = load('./test_data/BraTS20_Training_001_test1/SimTumor_warped_labels2.mha')
labels_output = np.zeros_like(labels)
#labels_output[np.where(labels==5)] = 3

edma_prob, _ = load('./test_data/BraTS20_Training_001_test1/SimTumor_prob4.mha')
core_prob, _ = load('./test_data/BraTS20_Training_001_test1/SimTumor_prob5.mha')

edma_vol = np.zeros_like(labels)
edma_vol[np.where(((edma_prob + core_prob) / 65536)>=0.5)] = 1
edma_vol = 1 * scipy.ndimage.binary_closing(edma_vol, structure=np.ones((10, 10, 10)))
edma_vol = 1 * scipy.ndimage.binary_fill_holes(edma_vol)
tumor_vol = np.zeros_like(labels)
tumor_vol[np.where((core_prob / 65536)>=0.5)] = 1
tumor_vol = 1 * scipy.ndimage.binary_fill_holes(tumor_vol)

labels_output[np.where(edma_vol==1)] = 1
labels_output[np.where(tumor_vol==1)] = 3
save(labels_output, './test_data/BraTS20_Training_001_test1/tumor_segmentation.nii.gz')