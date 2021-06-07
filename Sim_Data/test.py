from medpy.io import load
import trimesh
import meshio
import pymesh
file = '/home/mjia/Researches/Volume_Segmentation/NITRC-multi-file-downloads/TumorSimInput1/mesh_.vtk'
file1 = '/home/mjia/Researches/Volume_Segmentation/mindboggle/mindboggle_manually_labeled_individual_brains/Extra-18_surfaces/HLN-12-4/lh.labels.DKT25.manual.vtk'

mesh = meshio.read(file)
mesh.show()