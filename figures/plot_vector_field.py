import matplotlib.pyplot as plt
import numpy as np
from medpy.io import save, load
from PIL import Image

X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

#displacement, _ = load('/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/predicted_disps/' + str(335).zfill(5) + '.mha')
displacement, _ = load('/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/my_training_data/00205/SimTumor_def.mha')
displacement_slice = displacement[::5,::5,56,:]
displacement_slice = displacement_slice[10:40, 15:45, :]

X = np.arange(0, displacement_slice.shape[0], 1)
Y = np.arange(0, displacement_slice.shape[1], 1)
U = displacement_slice[:,:,1]
V = displacement_slice[:,:,0]

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, scale=400)
#ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#             label='Quiver key, length = 10', labelpos='E')


plt.savefig('vector_field_inv.png', dpi=300)
plt.show()