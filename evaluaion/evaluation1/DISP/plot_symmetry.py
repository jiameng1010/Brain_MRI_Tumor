# Python program to illustrate
# boxplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

def collecting_data(data_frame, text_f, tau, is_disp):
    f = open(text_f)
    lines = f.readlines()
    dice_org = float(lines[0])
    dice = float(lines[1])
    dif = dice - dice_org
    persentage = dif / dice_org
    return {'dif': dice,
            'persentage': persentage,
            'tau': tau,
            'network type': is_disp}

# use to set style of background of plot
outputdir1 = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/evaluaion/evaluation1/DISP/'
outputdir2 = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/evaluaion/evaluation1/SVF/'
my_data = pd.DataFrame(columns = ['dif', 'persentage', 'tau', 'network type'])
for ii in range(10):
    dis_scale = ii/10
    dir = outputdir1+str(ii)
    for entry in os.scandir(dir):
        data = collecting_data(my_data, entry.path, dis_scale, is_disp='DISP')
        my_data = my_data.append(data, ignore_index=True)
for ii in range(10):
    dis_scale = ii/10
    dir = outputdir2+str(ii)
    for entry in os.scandir(dir):
        data = collecting_data(my_data, entry.path, dis_scale, is_disp='SVF')
        my_data = my_data.append(data, ignore_index=True)

plt.grid(axis='y')
#plt.figure(figsize=[7, 3])
ax = seaborn.boxplot(x="tau",
                y="persentage",
                hue="network type",
                data=my_data)
plt.ylim([-0.08, 0.12])
plt.ylabel('Relative Symmtry Score Improvement')
plt.xlabel(r'$\tau$')
#plt.show()
plt.savefig('/home/mjia/Dropbox/papers/MICCAI2022/Figures/Fig_symmetry/symmetry_improve.png')
plt.clf()
#plt.figure(figsize=[7, 3])

plt.grid(axis='y')
seaborn.boxplot(x="tau",
                y="dif",
                hue="network type",
                data=my_data)
plt.ylim([0.3, 0.75])
plt.ylabel('Symmtry Score')
plt.xlabel(r'$\tau$')
#plt.show()
plt.savefig('/home/mjia/Dropbox/papers/MICCAI2022/Figures/Fig_symmetry/symmetry.png')