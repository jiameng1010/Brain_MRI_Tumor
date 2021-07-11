import igl
import trimesh
import numpy as np
#import open3d as o3d
import copy
import pymesh
from functools import reduce
from scipy import sparse
import h5py
from scipy.spatial import Voronoi, Delaunay

num_of_combination = np.asarray([2, 3, 4, 5, 6, 7, 8])

def zero_padding(volume):
    shape0 = volume.shape
    zeros = np.zeros(shape=(shape0[0]+2, shape0[1]+2, shape0[2]+2))
    zeros[1:-1, 1:-1, 1:-1] = volume
    return zeros

def apply_rigid_transform(S, T, R, tet_v):
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr

def affine(affine_m, points):
    point_aug = np.concatenate([points, np.ones(shape=(points.shape[0], 1), dtype=points.dtype)], axis=1)
    point_transformed = affine_m.dot(point_aug.T).T[:, :3]
    return point_transformed


def clap_voxels_out1(voxels_in_displacement, shape):
    valid_voxels0 = np.logical_and(voxels_in_displacement[:, 0] > 0, voxels_in_displacement[:, 0] < (shape[0]-1))
    valid_voxels1 = np.logical_and(voxels_in_displacement[:, 1] > 0, voxels_in_displacement[:, 1] < (shape[1]-1))
    valid_voxels2 = np.logical_and(voxels_in_displacement[:, 2] > 0, voxels_in_displacement[:, 2] < (shape[2]-1))
    valid_voxels = 1 * np.logical_and(np.logical_and(valid_voxels2, valid_voxels1), valid_voxels0)
    voxels_in_displacement = voxels_in_displacement[np.where(valid_voxels == 1)]
    return voxels_in_displacement

def clap_voxels_out(voxels_in_displacement, index, shape):
    valid_voxels0 = np.logical_and(voxels_in_displacement[:, 0] > 0, voxels_in_displacement[:, 0] < (shape[0]-1))
    valid_voxels1 = np.logical_and(voxels_in_displacement[:, 1] > 0, voxels_in_displacement[:, 1] < (shape[1]-1))
    valid_voxels2 = np.logical_and(voxels_in_displacement[:, 2] > 0, voxels_in_displacement[:, 2] < (shape[2]-1))
    valid_voxels = 1 * np.logical_and(np.logical_and(valid_voxels2, valid_voxels1), valid_voxels0)
    voxels_in_displacement = voxels_in_displacement[np.where(valid_voxels == 1)]
    return voxels_in_displacement, (index[0][np.where(valid_voxels == 1)], index[1][np.where(valid_voxels == 1)], index[2][np.where(valid_voxels == 1)])

def interpolate_without_displacement(org_data, affine_matrix, shape0):
    from scipy.interpolate import RegularGridInterpolator
    output_volume = np.zeros(shape=shape0, dtype=np.uint16)
    index = np.where(output_volume == 0)
    voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                         np.expand_dims(index[1], axis=1),
                                         np.expand_dims(index[2], axis=1)], axis=1)
    voxels_moved = affine(affine_matrix, voxels_interpolate)
    shape = org_data.shape
    x = np.linspace(0, shape[0]-1, shape[0])
    y = np.linspace(0, shape[1]-1, shape[1])
    z = np.linspace(0, shape[2]-1, shape[2])
    fn = RegularGridInterpolator((x, y, z), org_data)
    voxels_in_displacement, index = clap_voxels_out(voxels_moved, index, shape)
    values = fn(voxels_in_displacement)
    output_volume[index] = values
    return output_volume

def cascade_diaplacements(displacement_1, displacement_2):
    from scipy.interpolate import RegularGridInterpolator
    displacement_out = np.zeros_like(displacement_1)
    index = np.where(displacement_out[:,:,:,0] >= 0)
    voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                         np.expand_dims(index[1], axis=1),
                                         np.expand_dims(index[2], axis=1)], axis=1)
    shape = displacement_1.shape
    x = np.linspace(0, shape[0]-1, shape[0])
    y = np.linspace(0, shape[1]-1, shape[1])
    z = np.linspace(0, shape[2]-1, shape[2])
    #fn_1 = RegularGridInterpolator((x, y, z), displacement_1)
    fn_2 = RegularGridInterpolator((x, y, z), displacement_2)

    d1 = voxels_interpolate + np.reshape(displacement_1, (-1, 3))
    d1, index = clap_voxels_out(d1, index, shape)
    voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                         np.expand_dims(index[1], axis=1),
                                         np.expand_dims(index[2], axis=1)], axis=1)
    d2 = fn_2(d1) + d1
    displacement_out[index] = d2 - voxels_interpolate
    return displacement_out


def interpolate_with_displacement(org_data, label_manual_aseg, label_manual, displacement_volume, tumor_labels, affine_matrix):
    from scipy.interpolate import RegularGridInterpolator
    index = np.where(org_data >= 0)
    voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                         np.expand_dims(index[1], axis=1),
                                         np.expand_dims(index[2], axis=1)], axis=1)
    voxels_in_displacement = affine(affine_matrix, voxels_interpolate)
    shape = displacement_volume.shape
    x = np.linspace(0, shape[0]-1, shape[0])
    y = np.linspace(0, shape[1]-1, shape[1])
    z = np.linspace(0, shape[2]-1, shape[2])

    fn = RegularGridInterpolator((x, y, z), displacement_volume)
    voxels_in_displacement, index = clap_voxels_out(voxels_in_displacement, index, shape)
    displacements = fn(voxels_in_displacement)
    voxels_in_displacement_moved = voxels_in_displacement + displacements
    fn_Sim_label = RegularGridInterpolator((x, y, z), tumor_labels, method='nearest')
    voxels_moved = affine(np.linalg.inv(affine_matrix), voxels_in_displacement_moved)
    shape = org_data.shape
    x = np.linspace(0, shape[0]-1, shape[0])
    y = np.linspace(0, shape[1]-1, shape[1])
    z = np.linspace(0, shape[2]-1, shape[2])
    fn = RegularGridInterpolator((x, y, z), org_data)
    voxels_moved = np.concatenate([np.expand_dims(np.clip(voxels_moved[:, 0], 0.0, shape[0]), axis=1),
                                    np.expand_dims(np.clip(voxels_moved[:, 1], 0.0, shape[1]), axis=1),
                                    np.expand_dims(np.clip(voxels_moved[:, 2], 0.0, shape[2]), axis=1)], axis=1)
    value = fn(voxels_moved)
    out_data = np.zeros_like(org_data)
    out_data[index] = value

    fn = RegularGridInterpolator((x, y, z), label_manual, method='nearest')
    value = fn(voxels_moved)
    out_label_manual = np.zeros_like(label_manual)
    out_label_manual[index] = value

    fn = RegularGridInterpolator((x, y, z), label_manual_aseg, method='nearest')
    value = fn(voxels_moved)
    out_label_manual_aseg = np.zeros_like(label_manual_aseg)
    out_label_manual_aseg[index] = value

    value = fn_Sim_label(voxels_in_displacement)
    tumor_voxels = np.where(value == 5)
    a = np.min(out_label_manual_aseg)

    tumor_mask = np.zeros_like(out_label_manual_aseg)
    tumor_mask[(index[0][tumor_voxels], index[1][tumor_voxels], index[2][tumor_voxels])] = 1
    from scipy import ndimage
    tumor_mask = ndimage.binary_closing(tumor_mask).astype(np.int)
    #tumor_mask = ndimage.binary_closing(tumor_mask).astype(np.int)
    out_label_manual_aseg[np.where(tumor_mask == 1)] = -1

    return out_data, out_label_manual_aseg, out_label_manual

def apply_transform_non_rigid(S, T, R, C, tet_v, templates, d=0):
    #apply deformation
    if not isinstance(d, np.ndarray):
        deformation = np.reshape(np.matmul(templates, C), [-1, 3])
    else:
        deformation = d
    tet_v = tet_v + deformation
    #apply rotation
    #r = Rotation.from_rotvec(R).as_matrix()
    r1, r2, r3, rr = angle2rotmatrix(R)
    tran_v = np.matmul(rr, tet_v.T).T
    # apply scale
    tran_v = S * tran_v
    #apply translation
    tran_v = tran_v + np.expand_dims(T, axis=0)
    return tran_v, rr, deformation

def apply_transform_rigid_reverse(S, T, R, tet_v):
    r1, r2, r3, rr = angle2rotmatrix(R)
    invert_rr = np.linalg.inv(rr)
    tran_v = tet_v - np.expand_dims(T, axis=0)
    tran_v = tran_v / S
    tran_v = np.matmul(invert_rr, tran_v.T).T
    return tran_v, invert_rr

def size_tetra3(v1, v2, v3):
    xa = v1[:, 0]
    ya = v1[:, 1]
    za = v1[:, 2]
    xb = v2[:, 0]
    yb = v2[:, 1]
    zb = v2[:, 2]
    xc = v3[:, 0]
    yc = v3[:, 1]
    zc = v3[:, 2]

    result = xa*(yb*zc - zb*yc) - ya*(xb*zc - zb*xc) + za*(xb*yc - yb*xc)
    return np.abs(result)

def size_tetra3_real(v1, v2, v3):
    xa = v1[:, 0]
    ya = v1[:, 1]
    za = v1[:, 2]
    xb = v2[:, 0]
    yb = v2[:, 1]
    zb = v2[:, 2]
    xc = v3[:, 0]
    yc = v3[:, 1]
    zc = v3[:, 2]

    result = xa*(yb*zc - zb*yc) - ya*(xb*zc - zb*xc) + za*(xb*yc - yb*xc)
    return result

def tetra_interpolation_delaunay(feed_points, v_points, v_value):
    delaunay_tri = Delaunay(v_points)
    result = np.ones(shape=(feed_points.shape[0], v_value.shape[1]))
    simplex = delaunay_tri.find_simplex(feed_points)
    points_inside_trihull = np.where(simplex != -1)[0]
    points_not_inside_trihull = np.where(simplex == -1)[0]
    corresponding_simplices = simplex[points_inside_trihull]
    corresponding_feed_points = feed_points[points_inside_trihull, :]
    #result[points_inside_trihull] = 0

    # form tetrahedron
    vertices_1 = delaunay_tri.simplices[corresponding_simplices, 0]
    vertices_2 = delaunay_tri.simplices[corresponding_simplices, 1]
    vertices_3 = delaunay_tri.simplices[corresponding_simplices, 2]
    vertices_4 = delaunay_tri.simplices[corresponding_simplices, 3]
    '''p2p_distance = square_distance(feed_points, v_points)
    vertices_1 = np.argmin(p2p_distance, axis=1)
    p2p_distance[np.arange(p2p_distance.shape[0]), vertices_1] = np.Inf
    vertices_2 = np.argmin(p2p_distance, axis=1)
    p2p_distance[np.arange(p2p_distance.shape[0]), vertices_2] = np.Inf
    vertices_3 = np.argmin(p2p_distance, axis=1)
    p2p_distance[np.arange(p2p_distance.shape[0]), vertices_3] = np.Inf
    vertices_4 = np.argmin(p2p_distance, axis=1)'''

    v1 = v_points[vertices_1, :] - corresponding_feed_points
    v2 = v_points[vertices_2, :] - corresponding_feed_points
    v3 = v_points[vertices_3, :] - corresponding_feed_points
    v4 = v_points[vertices_4, :] - corresponding_feed_points

    #volume of sub-tetra
    Sp123 = np.expand_dims(size_tetra3(v1, v2, v3), axis=1)
    Sp234 = np.expand_dims(size_tetra3(v2, v3, v4), axis=1)
    Sp124 = np.expand_dims(size_tetra3(v1, v2, v4), axis=1)
    Sp134 = np.expand_dims(size_tetra3(v1, v3, v4), axis=1)

    interpolated = Sp123 * v_value[vertices_4] + Sp234 * v_value[vertices_1] + Sp124 * v_value[vertices_3] + Sp134 * v_value[vertices_2]
    interpolated = interpolated / (Sp123 + Sp234 + Sp124 + Sp134)
    result[points_inside_trihull, :] = interpolated

    return result, points_not_inside_trihull

def tetra_interpolation_full(feed_points_all, v_points, v_value):
    batch_points = 20000
    num_of_querry = feed_points_all.shape[0]
    num_of_runs = int(num_of_querry / batch_points)
    pred_all_value = np.zeros(shape=[num_of_querry, 3])
    # tri = Delaunay(self.voronoi_sample_points)
    for i in range(num_of_runs + 1):
        if i == num_of_runs:
            feed_points = feed_points_all[(i * batch_points):, :]
        else:
            feed_points = feed_points_all[(i * batch_points):((i + 1) * batch_points), :]

        pred = tetra_interpolation(feed_points, v_points, v_value)

        if i == 0:
            pred_all_value = pred
        else:
            pred_all_value = np.concatenate([pred_all_value, pred])
    return pred_all_value

def tetra_interpolation(feed_points, v_points, v_value):
    _, closet_vert_target = closest4(feed_points, v_points)
    vertices_1 = v_points[closet_vert_target[0, :], :]
    vertices_2 = v_points[closet_vert_target[1, :], :]
    vertices_3 = v_points[closet_vert_target[2, :], :]
    vertices_4 = v_points[closet_vert_target[3, :], :]
    corresponding_feed_points = feed_points
    v1 = vertices_1 - corresponding_feed_points
    v2 = vertices_2 - corresponding_feed_points
    v3 = vertices_3 - corresponding_feed_points
    v4 = vertices_4 - corresponding_feed_points

    #volume of sub-tetra
    Sp123 = np.expand_dims(size_tetra3(v1, v2, v3), axis=1)
    Sp234 = np.expand_dims(size_tetra3(v2, v3, v4), axis=1)
    Sp124 = np.expand_dims(size_tetra3(v1, v2, v4), axis=1)
    Sp134 = np.expand_dims(size_tetra3(v1, v3, v4), axis=1)

    interpolated = Sp123 * v_value[closet_vert_target[3, :], :]\
                   + Sp234 * v_value[closet_vert_target[0, :], :]\
                   + Sp124 * v_value[closet_vert_target[2, :], :]\
                   + Sp134 * v_value[closet_vert_target[1, :], :]
    interpolated = interpolated / (Sp123 + Sp234 + Sp124 + Sp134)
    return interpolated

def normalize(normal_vectors):
    norm = np.sum(normal_vectors*normal_vectors, axis=1)
    norm = np.sqrt(norm)
    norm = np.expand_dims(norm, axis=1)
    return (1.0/norm) * normal_vectors

def volume_one_tetrahedron(tetr_mesh, i):
    return 0


def mesh_volume(mesh):
    v0 = mesh.vertices[mesh.voxels[:, 0],:]
    v1 = mesh.vertices[mesh.voxels[:, 1], :]
    v2 = mesh.vertices[mesh.voxels[:, 2], :]
    v3 = mesh.vertices[mesh.voxels[:, 3], :]
    tetra_volume = size_tetra(v0, v1, v2, v3)
    return np.sum(tetra_volume)

def size_tetra(v0, v1, v2, v3):
    v11 = v1 - v0
    v22 = v2 - v0
    v33 = v3 - v0
    xa = v11[:, 0]
    ya = v11[:, 1]
    za = v11[:, 2]
    xb = v22[:, 0]
    yb = v22[:, 1]
    zb = v22[:, 2]
    xc = v33[:, 0]
    yc = v33[:, 1]
    zc = v33[:, 2]

    result = xa*(yb*zc - zb*yc) - ya*(xb*zc - zb*xc) + za*(xb*yc - yb*xc)
    return np.abs(result/6)

def parse(para):
    return para[0], para[1:4], para[4:7]

def parse_non_rigid(para):
    return para[0], para[1:4], para[4:7], para[7:]

def angle2rotmatrix(angle):
    r1 = np.asarray([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                     [np.sin(angle[2]), np.cos(angle[2]), 0],
                     [0, 0, 1]])
    r2 = np.asarray([[np.cos(angle[1]), 0, np.sin(angle[1])],
                     [0.0, 1.0, 0.0],
                     [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    r3 = np.asarray([[1, 0, 0],
                     [0, np.cos(angle[0]), -np.sin(angle[0])],
                     [0, np.sin(angle[0]), np.cos(angle[0])]])
    return r1, r2, r3, np.matmul(r1, np.matmul(r2, r3))

def angle2drotmatrix(angle):
    r1 = np.asarray([[-np.sin(angle[2]), -np.cos(angle[2]), 0],
                     [np.cos(angle[2]), -np.sin(angle[2]), 0],
                     [0, 0, 0.0]])
    r2 = np.asarray([[-np.sin(angle[1]), 0, np.cos(angle[1])],
                     [0.0, 0.0, 0.0],
                     [-np.cos(angle[1]), 0, -np.sin(angle[1])]])
    r3 = np.asarray([[0, 0, 0],
                     [0, -np.sin(angle[0]), -np.cos(angle[0])],
                     [0, np.cos(angle[0]), -np.sin(angle[0])]])
    return r1, r2, r3

def load_PC(data_dir):
    FF = data_dir + '/' + data_dir.split('/')[-1] + "FF.xyz"
    LR = data_dir + '/' + data_dir.split('/')[-1] + "LR.xyz"
    RR = data_dir + '/' + data_dir.split('/')[-1] + "RR.xyz"
    SR = data_dir + '/' + data_dir.split('/')[-1] + "SR.xyz"
    FF_f = open(FF, 'r')
    LR_f = open(LR, 'r')
    RR_f = open(RR, 'r')
    SR_f = open(SR, 'r')
    FF_str = FF_f.readlines()
    LR_str = LR_f.readlines()
    RR_str = RR_f.readlines()
    SR_str = SR_f.readlines()
    FF_f.close()
    LR_f.close()
    RR_f.close()
    SR_f.close()

    PC_str = FF_str + LR_str + RR_str + SR_str
    Points = []
    for v in PC_str:
        vv = np.array(v.split(' ')).astype(np.float)[1:]
        Points.append(vv)
    point_cloud = np.array(Points)

    label = np.concatenate([2 * np.ones(shape=len(FF_str), dtype=np.int16),
                            3 * np.ones(shape=len(LR_str), dtype=np.int16),
                            4 * np.ones(shape=len(RR_str), dtype=np.int16),
                            1 * np.ones(shape=len(SR_str), dtype=np.int16),], axis=0)

    return point_cloud, label

def init_reg(point_cloud, PC_lable, tet_v, meshes):
    target0 = point_cloud[np.where(PC_lable==0)[0], :]
    pc_target0 = o3d.geometry.PointCloud()
    pc_target0.points = o3d.utility.Vector3dVector(target0)
    pc_target0.paint_uniform_color([1,0,0])
    target1 = point_cloud[np.where(PC_lable==1)[0], :]
    pc_target1 = o3d.geometry.PointCloud()
    pc_target1.points = o3d.utility.Vector3dVector(target1)
    pc_target1.paint_uniform_color([0,1,0])
    target2 = point_cloud[np.where(PC_lable==2)[0], :]
    pc_target2 = o3d.geometry.PointCloud()
    pc_target2.points = o3d.utility.Vector3dVector(target2)
    pc_target2.paint_uniform_color([0,0,1])
    pc_target = pc_target0 + pc_target1 + pc_target2
    pc_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=5000))

    source = tet_v[np.unique(np.resize(meshes[0].faces,(-1))), :]
    pc_source0 = o3d.geometry.PointCloud()
    pc_source0.points = o3d.utility.Vector3dVector(source)
    pc_source0.paint_uniform_color([1,0,0])
    source = tet_v[np.unique(np.resize(meshes[1].faces,(-1))), :]
    pc_source1 = o3d.geometry.PointCloud()
    pc_source1.points = o3d.utility.Vector3dVector(source)
    pc_source1.paint_uniform_color([0,1,0])
    source = tet_v[np.unique(np.resize(meshes[2].faces,(-1))), :]
    pc_source2 = o3d.geometry.PointCloud()
    pc_source2.points = o3d.utility.Vector3dVector(source)
    pc_source2.paint_uniform_color([0,0,1])
    pc_source = pc_source0 + pc_source1 + pc_source2
    pc_source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=5000))

    print("Apply point-to-point ICP")
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    reg_p2p = o3d.registration.registration_colored_icp(
        pc_source, pc_target, 50, trans_init, o3d.registration.ICPConvergenceCriteria(relative_fitness=1,
                                                    relative_rmse=1,
                                                    max_iteration=20))
    '''reg_p2p = o3d.registration.registration_icp(
        pc_target, pc_source, 3, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())'''
    draw_registration_result(pc_source, pc_target, reg_p2p.transformation)
    return np.asarray([1]), reg_p2p.transformation[0:3, 3], reg_p2p.transformation[0:3, 0:3]

def load_voxel_mesh():
    # load the liver mesh and save it to OFF file
    node_f = open('../../org/LiverVolume.nod', 'r')
    tetr_f = open('../../org/LiverVolume.elm', 'r')
    face_f = open('../../org/LiverVolume.bel', 'r')

    vertices_str = node_f.readlines()
    faces_str = face_f.readlines()
    tetr_str = tetr_f.readlines()
    vertices = []
    for v in vertices_str:
        vv = np.array(v.split(' ')).astype(np.float)[1:]
        vertices.append(vv)
    vertices = np.array(vertices)

    faces = []
    for f in faces_str:
        ff = np.array(f.split(' ')).astype(np.int)[1:] - 1
        faces.append(ff)
    faces = np.array(faces)
    tmp = copy.copy(faces[:, 0])
    faces[:, 0] = faces[:, 1]
    faces[:, 1] = tmp

    voxels = []
    for f in tetr_str:
        ff = np.array(f.split(' ')).astype(np.int)[1:] - 1
        voxels.append(ff)
    voxels = np.array(voxels)

    # mesh_liver = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh_liver.show()
    # mesh_liver.export('../org/Liver.off')

    mesh_liver = pymesh.form_mesh(vertices=vertices, faces=faces, voxels=voxels)
    return mesh_liver

def load_disp_solutions(dir, num):
    solutions = []
    for i in range(num):
        with h5py.File(dir + '/' + str(i).zfill(3) + '.h5', 'r') as hf:
            coefficient = hf['displacement'][:]
            coefficient = np.reshape(coefficient, [-1])
            solutions.append(np.expand_dims(coefficient, 1))
    return np.concatenate(solutions, axis=1)

def re_order_templates():
    templates = load_disp_solutions('../displacement_solutions/Y1000000.0_P0.49_14p', 14*3)
    templates = np.reshape(templates, [-1, 3, 14*3])
    with h5py.File('../V_order.h5', 'r') as hf:
        order = hf['order'][:]
    #templates = templates[order, :, :]
    templates = np.reshape(templates, [-1, 14*3])
    return templates


def closest(A, B):
    distance = np.expand_dims(A, axis=0) - np.expand_dims(B, axis=1)
    distance = np.sum(distance * distance, axis=2)
    closest = np.argmin(distance, axis=0)
    return distance[closest, np.arange(A.shape[0])], closest

def closest4(A, B):
    distance = np.expand_dims(A, axis=0) - np.expand_dims(B, axis=1)
    distance = np.sum(distance * distance, axis=2)
    closest = np.argsort(distance, axis=0)[:4]
    return distance[closest, np.arange(A.shape[0])], closest

def load_xyz(filename):
    with open(filename) as f:
        points_str = f.readlines()
    vertices = []
    for v in points_str:
        vv = np.array(v.split(' ')).astype(np.float)[1:]
        vertices.append(vv)
    vertices = np.array(vertices)
    return vertices


def load_output_xyz(filename):
    with open(filename) as f:
        points_str = f.readlines()
    vertices = []
    for v in points_str:
        vv = np.array(v.split(' ')).astype(np.float)[4:]
        vertices.append(vv)
    vertices = np.array(vertices)
    return vertices

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def parse_source_info(filename):
    f = open(filename, 'r')
    text = f.readlines()
    f.close()
    return int(text[1].split(' ')[-1]), int(text[2].split(' ')[-1]), int(text[3].split(' ')[-1])

def load_affine_transform(file_name, start_from=8):
    f = open(file_name)
    text = f.readlines()
    affine_matrix = np.eye(4)
    strings = text[start_from-1].split(' ')
    strings_out = []
    for i in strings:
        if i != '':
            strings_out.append(i)
    affine_matrix[0, :] = np.asarray(
        [float(strings_out[0]), float(strings_out[1]), float(strings_out[2]),
         float(strings_out[3])])

    strings = text[start_from].split(' ')
    strings_out = []
    for i in strings:
        if i != '':
            strings_out.append(i)
    affine_matrix[2, :] = np.asarray(
        [float(strings_out[0]), float(strings_out[1]), float(strings_out[2]),
         float(strings_out[3])])

    strings = text[start_from+1].split(' ')
    strings_out = []
    for i in strings:
        if i != '':
            if i.endswith(';\n'):
                strings_out.append(i[:-2])
            else:
                strings_out.append(i)
    affine_matrix[1, :] = np.asarray(
        [float(strings_out[0]), float(strings_out[1]), float(strings_out[2]),
         float(strings_out[3])])
    return affine_matrix