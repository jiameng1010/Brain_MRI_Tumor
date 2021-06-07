import pymesh
import numpy as np
import scipy
import copy
import igl
#from plot import plot_points, plot_mesh, plot_mesh_points, plot_mesh_points_label,\
#    plot_meshes, plot_mesh_force, compute_n_plot_force, plot_mesh_points_vector
#from debug_plot import plot_displacement, plot_mesh_surfacenormal
#from util_package.util import load_voxel_mesh
#from scikits.umfpack import spsolve

num_of_combination = np.asarray([1,2,3,4,5,6,7,8])
order_of_polynomial = int(3)

def get_displacement_boundary(C_matrix, vertices, pca):
    x = np.matmul(vertices, pca[:,0])
    y = np.matmul(vertices, pca[:,1])
    d = np.polynomial.polynomial.polyval2d(x,y, 10*C_matrix)
    displacement = np.tile(np.expand_dims(d, axis=1), [1,3]) * pca[:,2]
    displacement = displacement - np.mean(displacement, axis=0, keepdims=True)
    #plot_displacement(mesh_back, displacement)
    return displacement

def sparse_pinv(left_Matrix_s):
    u, s, v = scipy.sparse.linalg.svds(left_Matrix_s, k=int(left_Matrix_s.shape[0]/2))
    s_inv = np.diag(1/s)
    K11E_inv_svd = np.matmul(v.T, np.matmul(s_inv, u.T))
    return K11E_inv_svd


def solve_one(C, mesh, stiffness, back_index, pca_vectors):
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[back_index] = 1
    order = np.argsort(back_indicator)
    order3 = np.reshape(np.concatenate([np.expand_dims(order, axis=1),
                         np.expand_dims(order+1, axis=1),
                         np.expand_dims(order+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator))
    num_of_vertices = mesh.vertices.shape[0]

    C_matrix = np.zeros(shape=(order_of_polynomial+1, order_of_polynomial+1))
    iu1 = np.triu_indices(4)
    C_matrix[iu1] = C
    C_matrix = np.fliplr(C_matrix)

    vertices_sorted = copy.copy(mesh.vertices[order,:])
    stiff_sorted = stiffness[:, order3]
    #plot_mesh_points(mesh.vertices, mesh.faces, vertices_sorted[(num_of_vertices-num_of_boundary):, :])
    displacement_boundary = 3*get_displacement_boundary(C_matrix, vertices_sorted[(num_of_vertices-num_of_boundary):,:], pca_vectors[1])
    #zero_pad_boundary = copy.copy(displacement_boundary)
    #zero_pad_boundary[np.where(back_indicator == 0), :] = 0
    #zero_pad_boundary = np.reshape(zero_pad_boundary, (-1))

    seg = 3*(num_of_vertices-num_of_boundary)
    K11 = stiff_sorted[:, :seg]
    K22 = stiff_sorted[:, seg:]
    K11E = np.concatenate([K11, -1*np.eye(K11.shape[0])], axis=1)
    left_Matrix_s = scipy.sparse.csr_matrix(K11E)
    u, s, v = scipy.sparse.linalg.svds(left_Matrix_s, k=400)
    s_inv = np.diag(1/s)
    K11E_inv = np.linalg.pinv(K11E)
    K11E_inv_svd = np.matmul(v.T, np.matmul(s_inv, u.T))
    dnF = np.matmul(K11E_inv, -np.matmul(K22, np.reshape(displacement_boundary, (-1))))
    free_displacement = np.reshape(dnF[:seg], (-1, 3))

    displacement = np.concatenate([free_displacement, displacement_boundary], axis=0)
    new_vertices_sorted = vertices_sorted + displacement
    new_vertices = new_vertices_sorted[order_reverse, :]
    plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces)
    plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    return new_vertices

def solve_one_laplacian(C, mesh, stiff, laplacian, back_index, pca_vectors):
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[back_index] = 1
    order = np.argsort(back_indicator)
    order3 = np.reshape(np.concatenate([np.expand_dims(order, axis=1),
                         np.expand_dims(order+1, axis=1),
                         np.expand_dims(order+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator))
    num_of_vertices = mesh.vertices.shape[0]

    C_matrix = np.zeros(shape=(order_of_polynomial+1, order_of_polynomial+1))
    iu1 = np.triu_indices(4)
    C_matrix[iu1] = 2*C
    C_matrix = np.fliplr(C_matrix)
    vertices_sorted = copy.copy(mesh.vertices[order,:])
    displacement_boundary = get_displacement_boundary(C_matrix, vertices_sorted[(num_of_vertices-num_of_boundary):,:], pca_vectors[1])


    laplacian_i = laplacian[order[:(num_of_vertices-num_of_boundary)], :]
    laplacian_ii = laplacian_i[:, order[:(num_of_vertices - num_of_boundary)]]
    laplacian_ib = laplacian_i[:, order[(num_of_vertices - num_of_boundary):]]
    #plot_mesh_points(mesh.vertices, mesh.faces, vertices_sorted[(num_of_vertices-num_of_boundary):, :])
    free_displacement = scipy.sparse.linalg.spsolve(laplacian_ii, laplacian_ib.dot(displacement_boundary))
    displacement = np.concatenate([-free_displacement, displacement_boundary], axis=0)
    force = stiff.dot(np.reshape(displacement, [-1]))
    force = np.reshape(force, [3, -1]).T
    force_mag = np.sum(force*force, axis=1)
    new_vertices_sorted = vertices_sorted + displacement
    new_vertices = new_vertices_sorted[order_reverse, :]
    #plot_mesh_force(new_vertices, mesh.faces, mesh.voxels, np.log(force_mag))
    plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces)
    plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    print(mesh.volume)
    print(mesh_deformed.volume)
    return mesh_deformed


def solve_one_stiffness(mesh, stiff, back_index, displacement):
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[back_index] = 1
    order = np.argsort(back_indicator)
    order_3 = 3*order
    order3 = np.reshape(np.concatenate([np.expand_dims(order_3, axis=1),
                         np.expand_dims(order_3+1, axis=1),
                         np.expand_dims(order_3+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator))
    num_of_vertices = mesh.vertices.shape[0]

    vertices_sorted = copy.copy(mesh.vertices[order,:])
    displacement_boundary = displacement[order[(num_of_vertices - num_of_boundary):], :]

    seg = 3 * (num_of_vertices - num_of_boundary)
    stiff_i = stiff[order3[:seg], :]
    stiff_ii = stiff_i[:, order3[:seg]]
    left = scipy.sparse.hstack([stiff_ii, scipy.sparse.eye(stiff_ii.shape[0])])
    stiff_ib = stiff_i[:, order3[seg:]]

    zero_displacement = copy.copy(displacement)
    zero_displacement[np.where(back_index!=1),:] = 0
    plot_mesh_points_vector(mesh.vertices, mesh.faces, vertices_sorted[(num_of_vertices - num_of_boundary):, :],
                            displacement_boundary)
    return
    #d = spsolve(left, stiff_ib.dot(np.reshape(displacement_boundary, (-1))),)

    #stiff_ii_inv = sparse_pinv(left)
    #d = np.matmul(stiff_ii_inv, stiff_ib.dot(np.reshape(displacement_boundary, (-1))))
    d = scipy.sparse.linalg.lsmr(left, stiff_ib.dot(np.reshape(displacement_boundary, (-1))))


    free_displacement = np.reshape(d[0][:seg], (-1, 3))
    displacement = np.concatenate([-free_displacement, displacement_boundary], axis=0)
    new_vertices_sorted = vertices_sorted + displacement
    new_vertices = new_vertices_sorted[order_reverse, :]
    plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces, back_index)
    plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    compute_n_plot_force(mesh_deformed, stiff, (mesh_deformed.vertices-mesh.vertices))
    print(mesh.volume)
    print(mesh_deformed.volume)
    return mesh_deformed

def expand(l_s):
    l = l_s.toarray()
    indptr = np.reshape(np.tile(np.expand_dims(l_s.indptr, axis=1), [1, 3]), (-1))
    l_pad1 = scipy.sparse.csc_matrix(
        (l_s.data, 3*l_s.indices, indptr[2:]),
        shape = (3*l_s.shape[0], 3*l_s.shape[0]))
    l_pad2 = scipy.sparse.csc_matrix(
        (l_s.data, 3*l_s.indices+np.ones_like(l_s.indices), indptr[1:-1]),
        shape = (3*l_s.shape[0], 3*l_s.shape[0]))
    l_pad3 = scipy.sparse.csc_matrix(
        (l_s.data, 3*l_s.indices+2*np.ones_like(l_s.indices), indptr[:-2]),
        shape = (3*l_s.shape[0], 3*l_s.shape[0]))
    return l_pad1+l_pad2+l_pad3


def solve_one_elastic(mesh, stiff, laplacian, back_index, displacement):
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[back_index] = 1
    order = np.argsort(back_indicator)
    order_3 = 3*order
    order3 = np.reshape(np.concatenate([np.expand_dims(order_3, axis=1),
                         np.expand_dims(order_3+1, axis=1),
                         np.expand_dims(order_3+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator))
    num_of_vertices = mesh.vertices.shape[0]

    vertices_sorted = copy.copy(mesh.vertices[order,:])
    displacement_boundary = displacement[order[(num_of_vertices-num_of_boundary):], :]

    seg = 3 * (num_of_vertices - num_of_boundary)
    stiff_i = stiff[order3[:seg],:]
    stiff_ii = stiff_i[:, order3[:seg]]
    stiff_ib = stiff_i[:, order3[seg:]]
    l = laplacian[order[:-num_of_boundary], :]
    l = l[:, order[:-num_of_boundary]]
    l_expand = expand(l)
    zeros = scipy.sparse.csc_matrix(
        (l_expand.shape[0], stiff_ii.shape[1]), dtype=np.float64)
    eye = scipy.sparse.eye(stiff_ii.shape[0],format="csc")
    left = scipy.sparse.vstack(
        (scipy.sparse.hstack((stiff_ii, eye)),
         scipy.sparse.hstack((zeros, l_expand))))
    right = np.concatenate(
        [stiff_ib.dot(np.reshape(displacement_boundary, (-1))),
         np.zeros(seg)], axis=0)
    d = scipy.sparse.linalg.lsmr(left, right)
    #left_inv = scipy.sparse.linalg.inv(left)
    #d = left_inv.dot(right)

    free_displacement = np.reshape(d[0][:seg], (-1, 3))
    displacement = np.concatenate([-free_displacement, displacement_boundary], axis=0)
    new_vertices_sorted = vertices_sorted + displacement
    new_vertices = new_vertices_sorted[order_reverse, :]
    plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces, back_index)
    #plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    compute_n_plot_force(mesh_deformed, stiff, (mesh_deformed.vertices-mesh.vertices))
    print(mesh.volume)
    print(mesh_deformed.volume)
    return mesh_deformed

def solve_one_elastic_new(mesh, stiff, laplacian, back_index, displacement):
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[back_index] = 1
    order = np.argsort(back_indicator)
    order_3 = 3*order
    order3 = np.reshape(np.concatenate([np.expand_dims(order_3, axis=1),
                         np.expand_dims(order_3+1, axis=1),
                         np.expand_dims(order_3+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator))
    num_of_vertices = mesh.vertices.shape[0]

    vertices_sorted = copy.copy(mesh.vertices[order,:])
    displacement_boundary = displacement[order[(num_of_vertices-num_of_boundary):], :]

    seg = 3 * (num_of_vertices - num_of_boundary)
    stiff_i = stiff[order3[:seg],:]
    stiff_ii = stiff_i[:, order3[:seg]]
    stiff_ib = stiff_i[:, order3[seg:]]
    left = stiff_ii
    right = stiff_ib.dot(np.reshape(displacement_boundary, (-1)))
    d = scipy.sparse.linalg.lsmr(left, right)
    #left_inv = scipy.sparse.linalg.inv(left)
    #d = left_inv.dot(right)

    free_displacement = np.reshape(d[0], (-1, 3))
    displacement = np.concatenate([-free_displacement, displacement_boundary], axis=0)
    new_vertices_sorted = vertices_sorted + displacement
    new_vertices = new_vertices_sorted[order_reverse, :]
    plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces, back_index)
    #plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    compute_n_plot_force(mesh_deformed, stiff, (mesh_deformed.vertices-mesh.vertices))
    print(mesh.volume)
    print(mesh_deformed.volume)
    return mesh_deformed


def solve_one_quadratic(mesh, stiff, laplacian, back_index, displacement):
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[back_index] = 1
    order = np.argsort(back_indicator)
    order_3 = 3*order
    order3 = np.reshape(np.concatenate([np.expand_dims(order_3, axis=1),
                         np.expand_dims(order_3+1, axis=1),
                         np.expand_dims(order_3+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator))
    num_of_vertices = mesh.vertices.shape[0]
    seg = 3 * (num_of_vertices - num_of_boundary)

    displacement_boundary = displacement[order[(num_of_vertices-num_of_boundary):], :]
    #Q = scipy.sparse.eye(stiff.shape[0],format="csc")
    Q = stiff.T.dot(stiff)
    B = np.zeros(shape=(stiff.shape[0]))
    b = order3[seg:]
    bc = np.reshape(displacement_boundary, [-1])
    #Aeq = stiff
    #Beq = stiff.dot(np.reshape(displacement,[-1]))
    Aeq = scipy.sparse.csc_matrix((0, 0))
    Beq = np.array([])
    #Aeq1 = scipy.sparse.csc_matrix((3*num_of_boundary-3, stiff.shape[0]))
    #Aeq1[np.arange(0,b.shape[0]-3), b[3:]] = 1.0
    #Aeq2 = scipy.sparse.csc_matrix((seg, stiff.shape[0]))
    #Aeq = scipy.sparse.vstack([Aeq1, Aeq2], format='csr')
    #Beq1 = np.zeros(shape=[seg])
    #Beq2 = bc[3:]
    #Beq = np.concatenate([Beq2, Beq1], axis=0)
    _, d = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, is_A_pd=True)

    free_displacement = np.reshape(d, (-1, 3))
    new_vertices = mesh.vertices + free_displacement
    #plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces, back_index)
    #plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    compute_n_plot_force(mesh_deformed, stiff, (mesh_deformed.vertices-mesh.vertices))
    print(mesh.volume)
    print('volume: ' + str(mesh_deformed.volume))
    print('***********')
    return mesh_deformed

def extract_surface_nodes(mesh):
    meshS = pymesh.form_mesh(mesh.vertices, mesh.faces)
    meshS, info = pymesh.remove_isolated_vertices(meshS)
    surface_nodes_orgindex = info['ori_vertex_index']
    node_is_onsurface = np.zeros(shape=(mesh.vertices.shape[0]), dtype=np.int8)
    node_is_onsurface[surface_nodes_orgindex] = 1
    return surface_nodes_orgindex, node_is_onsurface

def solve_one_quadratic_new(mesh, stiff, laplacian, back_index, displacement):
    surface_nodes_orgindex, node_is_onsurface = extract_surface_nodes(mesh)
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[surface_nodes_orgindex] = 1
    back_indicator[back_index] = 2
    order = np.argsort(back_indicator)
    order_3 = 3*order
    order3 = np.reshape(np.concatenate([np.expand_dims(order_3, axis=1),
                         np.expand_dims(order_3+1, axis=1),
                         np.expand_dims(order_3+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator==2))
    num_of_front_surface = int(sum(back_indicator==1))
    num_of_vertices = mesh.vertices.shape[0]

    seg = 3 * (num_of_vertices - num_of_boundary)
    seg_surface = 3 * (num_of_vertices - num_of_boundary - num_of_front_surface)
    stiff_high = stiff[order3[:seg_surface], :]
    stiff_low = stiff[order3[seg_surface:],:]
    displacement_boundary = displacement[order[(num_of_vertices-num_of_boundary):], :]
    #Q = scipy.sparse.eye(stiff.shape[0],format="csc")
    Q = stiff_low.T.dot(stiff_low).tocsc()
    B = np.zeros(shape=(stiff.shape[0]))
    b = order3[seg:]
    bc = np.reshape(displacement_boundary, [-1])
    #Aeq = stiff
    #Beq = stiff.dot(np.reshape(displacement,[-1]))
    Aeq = stiff_high
    Beq = np.zeros(shape=seg_surface)
    #Aeq1 = scipy.sparse.csc_matrix((3*num_of_boundary-3, stiff.shape[0]))
    #Aeq1[np.arange(0,b.shape[0]-3), b[3:]] = 1.0
    #Aeq2 = scipy.sparse.csc_matrix((seg, stiff.shape[0]))
    #Aeq = scipy.sparse.vstack([Aeq1, Aeq2], format='csr')
    #Beq1 = np.zeros(shape=[seg])
    #Beq2 = bc[3:]
    #Beq = np.concatenate([Beq2, Beq1], axis=0)
    _, d = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, is_A_pd=False)

    free_displacement = np.reshape(d, (-1, 3))
    new_vertices = mesh.vertices + free_displacement
    #plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces, back_index)
    #plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    #compute_n_plot_force(mesh_deformed, stiff, (mesh_deformed.vertices-mesh.vertices))
    print(mesh.volume)
    print('volume: ' + str(mesh_deformed.volume))
    print('***********')
    return mesh_deformed

def simplifying_mesh(mesh_org):
    return 0


def solve_one_quadratic_simplify(mesh, stiff, laplacian, back_index, displacement):
    mesh_org = pymesh.form_mesh(mesh.vertices, mesh.faces, mesh.voxels)
    mesh = simplifying_mesh(mesh)

    surface_nodes_orgindex, node_is_onsurface = extract_surface_nodes(mesh)
    back_indicator = np.zeros(shape=mesh.vertices.shape[0])
    back_indicator[surface_nodes_orgindex] = 1
    back_indicator[back_index] = 2
    order = np.argsort(back_indicator)
    order_3 = 3*order
    order3 = np.reshape(np.concatenate([np.expand_dims(order_3, axis=1),
                         np.expand_dims(order_3+1, axis=1),
                         np.expand_dims(order_3+2, axis=1)], axis=1), (-1))
    order_reverse = np.argsort(order)
    num_of_boundary = int(sum(back_indicator==2))
    num_of_front_surface = int(sum(back_indicator==1))
    num_of_vertices = mesh.vertices.shape[0]

    seg = 3 * (num_of_vertices - num_of_boundary)
    seg_surface = 3 * (num_of_vertices - num_of_boundary - num_of_front_surface)
    stiff_high = stiff[order3[:seg_surface], :]
    stiff_low = stiff[order3[seg_surface:],:]
    displacement_boundary = displacement[order[(num_of_vertices-num_of_boundary):], :]
    #Q = scipy.sparse.eye(stiff.shape[0],format="csc")
    Q = stiff_low.T.dot(stiff_low).tocsc()
    B = np.zeros(shape=(stiff.shape[0]))
    b = order3[seg:]
    bc = np.reshape(displacement_boundary, [-1])
    #Aeq = stiff
    #Beq = stiff.dot(np.reshape(displacement,[-1]))
    Aeq = stiff_high
    Beq = np.zeros(shape=seg_surface)
    #Aeq1 = scipy.sparse.csc_matrix((3*num_of_boundary-3, stiff.shape[0]))
    #Aeq1[np.arange(0,b.shape[0]-3), b[3:]] = 1.0
    #Aeq2 = scipy.sparse.csc_matrix((seg, stiff.shape[0]))
    #Aeq = scipy.sparse.vstack([Aeq1, Aeq2], format='csr')
    #Beq1 = np.zeros(shape=[seg])
    #Beq2 = bc[3:]
    #Beq = np.concatenate([Beq2, Beq1], axis=0)
    _, d = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, is_A_pd=True)

    free_displacement = np.reshape(d, (-1, 3))
    new_vertices = mesh.vertices + free_displacement
    #plot_meshes(mesh.vertices, mesh.faces, new_vertices, mesh.faces, back_index)
    #plot_mesh_points(new_vertices, mesh.faces, new_vertices)
    mesh_deformed = pymesh.form_mesh(new_vertices, mesh.faces, mesh.voxels)
    #compute_n_plot_force(mesh_deformed, stiff, (mesh_deformed.vertices-mesh.vertices))
    print(mesh.volume)
    print('volume: ' + str(mesh_deformed.volume))
    print('***********')
    return mesh_deformed