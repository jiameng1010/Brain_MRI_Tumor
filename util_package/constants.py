import numpy as np

###BraTH
BraTS_Affine = np.asarray([[-1.0, 0.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0, 239.0],
                           [0.0, 0.0, 1.12, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

theta = -20 * np.pi / 180
BraTS_Affine_rotationx = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, np.cos(theta), -np.sin(theta), 0.0],
                                     [0.0, np.sin(theta), np.cos(theta), 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])

BraTS_Affine = BraTS_Affine_rotationx.dot(BraTS_Affine)
BraTS_shift = np.asarray([-125.0, 130.0, 40.0])
BraTS_Affine[:3, 3] = BraTS_Affine[:3, 3] - BraTS_shift


### Mindboggle
Mindboggle_Affine_MNI152_org = np.asarray([[-1.0, 0.0, 0.0, 90.0],
                                       [0.0, 1.0, 0.0, -126.0],
                                       [0.0, 0.0, 1.0, -72.0],
                                       [0.0, 0.0, 0.0, 1.0]])
Mindboggle_Affine_MNI152 = np.asarray([[-1.0, 0.0, 0.0, 90.0],
                                       [0.0, 1.0, 0.0, -126.0],
                                       [0.0, 0.0, 1.0, -72.0],
                                       [0.0, 0.0, 0.0, 1.0]])
Mindboggle_shift = np.asarray([0.0, -20.0, 8.0])
Mindboggle_Affine_MNI152[:3, 3] = Mindboggle_Affine_MNI152[:3, 3] - Mindboggle_shift

Mindboggle_Affine = np.asarray([[1.0, 0.0, 0.0, -90.0],
                                [0.0, 1.0, 0.0, -126.0],
                                [0.0, 0.0, 1.0, -72.0],
                                [0.0, 0.0, 0.0, 1.0]])



### BrainSim
BrainSim_Affine = np.asarray([[-1.0, 0.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
#BrainSim_shift = np.asarray([-128.0, -128.0, 80.0])
#BrainSim_Affine[:3, 3] = BrainSim_Affine[:3, 3] - BrainSim_shift

Affine_uniform2mindboggle = np.asarray([[160.0, 0.0, 0.0, 48.0],
                                        [0.0, 200.0, 0.0, 28.0],
                                        [0.0, 0.0, 179.0, 1.0],
                                        [0.0, 0.0, 0.0, 1.0]])