from mayavi import mlab
import numpy as np

#points: n*3
def plot_points(points):
    mlab.points3d(points[:,0], points[:,1], points[:,2], scale_factor=1)
    mlab.show()

#points1: n*3
#points2: n*3
def plot_pointss(points1, points2):
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.points3d(points1[:, 0], points1[:, 2], points1[:, 1], scale_factor=0.01, color=(1, 0, 0))
    mlab.points3d(points2[:, 0], points2[:, 2], points2[:, 1], scale_factor=0.01, color=(0, 1, 0))
    mlab.show()


#points1: n*3
#points2: n*3
def plot_pointses(points1, points2):
    mlab.figure(bgcolor=(1, 1, 1))
    v_points1 = points1[np.where(points1[:,0]<=np.mean(points2, axis=0)[0])[0], :]
    v_points2 = points1[np.where(points1[:, 1] <= np.mean(points2, axis=0)[1])[0], :]
    v_points3 = points1[np.where(points1[:, 2] <= np.mean(points2, axis=0)[2])[0], :]
    v_points = np.concatenate([v_points1, v_points2, v_points3], axis=0)
    mlab.points3d(v_points[:, 0], v_points[:, 2], v_points[:, 1], scale_factor=0.005, color=(1, 0, 0))
    v_points1 = points2[np.where(points2[:,0]<=np.mean(points2, axis=0)[0])[0], :]
    v_points2 = points2[np.where(points2[:, 1] <= np.mean(points2, axis=0)[1])[0], :]
    v_points3 = points2[np.where(points2[:, 2] <= np.mean(points2, axis=0)[2])[0], :]
    v_points = np.concatenate([v_points1, v_points2, v_points3], axis=0)
    mlab.points3d(v_points[:, 0], v_points[:, 2], v_points[:, 1], scale_factor=0.005, color=(0, 1, 0))
    mlab.show()

def plot_mesh_vectors(mesh_v, mesh_f, points, displacemennt):
    mlab.quiver3d(points[:, 0], points[:, 2], points[:, 1], displacemennt[:, 0], displacemennt[:, 2], displacemennt[:, 1],
                  color=(1, 0.3, 0.3), mode='2ddash', line_width=4, scale_factor=0.7)
    mlab.triangular_mesh([vert[0] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         [vert[1] for vert in mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

#mesh_v: m*3
#mesh_f: k*3
#points: n*3
def plot_mesh_points_vector(mesh_v, mesh_f, points, displacemennt):
    scale = 1
    mlab.figure(bgcolor=(1, 1, 1))
    #mlab.points3d(mesh_v[:,0], mesh_v[:,2], mesh_v[:,1], scale_factor=0.006, color=(1,1,1))
    mlab.points3d(points[:,0], points[:,2], points[:,1], scale_factor=0.005, color=(1,1,1))
    npoints = points+displacemennt
    nnpoints = displacemennt
    mlab.points3d(npoints[:, 0], npoints[:, 2], npoints[:, 1], scale_factor=0.005, color=(1, 0.6, 0.6))
    mlab.quiver3d(points[:,0], points[:,2], points[:,1], nnpoints[:, 0], nnpoints[:, 2], nnpoints[:, 1],
                  color=(1, 0.3, 0.3),mode='2ddash', line_width=4, scale_factor=1)
    mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

#mesh_v: m*3
#mesh_f: k*3
#points: n*3
def plot_mesh_points(mesh_v, mesh_f, points):
    scale = 1
    #mlab.figure(bgcolor=(1, 1, 1))
    #mlab.points3d(mesh_v[:,0], mesh_v[:,2], mesh_v[:,1], scale_factor=0.006, color=(1,1,1))
    extent = np.max(np.max(points, axis=0) - np.min(points, axis=0))
    mlab.points3d(points[:,0], points[:,2], points[:,1], scale_factor=0.01*extent, color=(1,1,1))
    mlab.triangular_mesh([vert[0] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         [vert[1] for vert in mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

def plot_points_label(points, PC_label):
    scale = 1
    mlab.figure(bgcolor=(1, 1, 1))
    #f = mlab.figure()
    points_FF = points[np.where(PC_label==2), :][0]
    points_LR = points[np.where(PC_label==3), :][0]
    points_RR = points[np.where(PC_label==4), :][0]
    points_SR = points[np.where(PC_label==1), :][0]
    p1=mlab.points3d(points_FF[:,0], points_FF[:,2], points_FF[:,1], scale_factor=0.02, color=(1, 0, 0))
    p2=mlab.points3d(points_LR[:,0], points_LR[:,2], points_LR[:,1], scale_factor=0.02, color=(0, 1, 0))
    p3=mlab.points3d(points_RR[:,0], points_RR[:,2], points_RR[:,1], scale_factor=0.02, color=(0, 0, 1))
    p4=mlab.points3d(points_SR[:,0], points_SR[:,2], points_SR[:,1], scale_factor=0.02, color=(0.2, 1, 1))
    mlab.show()

def plot_mesh_n_points_label(mesh_v, meshes, points, PC_label):
    scale = 1
    mlab.figure(bgcolor=(1, 1, 1))
    #f = mlab.figure()
    points_FF = points[np.where(PC_label==2), :][0]
    points_LR = points[np.where(PC_label==3), :][0]
    points_RR = points[np.where(PC_label==4), :][0]
    points_SR = points[np.where(PC_label==1), :][0]
    p1=mlab.points3d(points_FF[:,1], points_FF[:,2], points_FF[:,0], scale_factor=0.005, color=(1, 0, 0))
    p2=mlab.points3d(points_LR[:,1], points_LR[:,2], points_LR[:,0], scale_factor=0.005, color=(0, 1, 0))
    p3=mlab.points3d(points_RR[:,1], points_RR[:,2], points_RR[:,0], scale_factor=0.005, color=(0, 0, 1))
    p4=mlab.points3d(points_SR[:,1], points_SR[:,2], points_SR[:,0], scale_factor=0.005, color=(0.2, 1, 1))
    s4=mlab.triangular_mesh([vert[1] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[0] for vert in scale*mesh_v],
                         meshes[3].faces,
                         opacity=0.7,
                         color=(0.8, 0.8, 0.8))
    s1=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[0].faces,
                         opacity=0.5,
                         color=(1, 0, 0))#FF
    s2=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[1].faces,
                         opacity=0.5,
                         color=(0, 1, 0))
    s3=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[2].faces,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

def plot_mesh_points_label(mesh_v, meshes, points, PC_label):
    scale = 1
    #f = mlab.figure()
    points_FF = points[np.where(PC_label==0), :][0]
    points_LR = points[np.where(PC_label==1), :][0]
    points_RR = points[np.where(PC_label==2), :][0]
    points_SR = points[np.where(PC_label==3), :][0]
    p1=mlab.points3d(points_FF[:,0], points_FF[:,2], points_FF[:,1], scale_factor=0.01, color=(1, 0, 0))
    p2=mlab.points3d(points_LR[:,0], points_LR[:,2], points_LR[:,1], scale_factor=0.01, color=(0, 1, 0))
    p3=mlab.points3d(points_RR[:,0], points_RR[:,2], points_RR[:,1], scale_factor=0.01, color=(0, 0, 1))
    p4=mlab.points3d(points_SR[:,0], points_SR[:,2], points_SR[:,1], scale_factor=0.01, color=(0, 1, 1))
    s1=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[0].faces,
                         opacity=0.5,
                         color=(1, 0, 0))#FF
    s2=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[1].faces,
                         opacity=0.5,
                         color=(0, 1, 0))
    s3=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[2].faces,
                         opacity=0.5,
                         color=(0, 0, 1))
    s4=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[3].faces,
                         opacity=0.5,
                         color=(0, 1, 1))
    return [p1, p2, p3, p4, s1, s2, s3, s4]

def plot_mesh_points_label_anim(mesh_v, meshes, points_FF, points_LR, points_RR, points_SR):
    scale = 1
    p1=mlab.points3d(points_FF[:,0], points_FF[:,2], points_FF[:,1], scale_factor=0.01, color=(1, 0, 0))
    p2=mlab.points3d(points_LR[:,0], points_LR[:,2], points_LR[:,1], scale_factor=0.01, color=(0, 1, 0))
    p3=mlab.points3d(points_RR[:,0], points_RR[:,2], points_RR[:,1], scale_factor=0.01, color=(0, 0, 1))
    p4=mlab.points3d(points_SR[:,0], points_SR[:,2], points_SR[:,1], scale_factor=0.01, color=(0, 1, 1))
    s1=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[0].faces,
                         opacity=0.5,
                         color=(1, 0, 0))#FF
    s2=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[1].faces,
                         opacity=0.5,
                         color=(0, 1, 0))
    s3=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[2].faces,
                         opacity=0.5,
                         color=(0, 0, 1))
    s4=mlab.triangular_mesh([vert[0] for vert in scale*mesh_v],
                         [vert[2] for vert in scale*mesh_v],
                         [vert[1] for vert in scale*mesh_v],
                         meshes[3].faces,
                         opacity=0.5,
                         color=(0, 1, 1))
    return [p1, p2, p3, p4, s1, s2, s3, s4]

def update_scene_objs_anim(objs, tran_v, points_FF, points_LR, points_RR, points_SR):
    objs[0].mlab_source.x = points_FF[:,0]
    objs[0].mlab_source.y = points_FF[:,2]
    objs[0].mlab_source.z = points_FF[:,1]
    objs[1].mlab_source.x = points_LR[:,0]
    objs[1].mlab_source.y = points_LR[:,2]
    objs[1].mlab_source.z = points_LR[:,1]
    objs[2].mlab_source.x = points_RR[:,0]
    objs[2].mlab_source.y = points_RR[:,2]
    objs[2].mlab_source.z = points_RR[:,1]
    objs[3].mlab_source.x = points_SR[:,0]
    objs[3].mlab_source.y = points_SR[:,2]
    objs[3].mlab_source.z = points_SR[:,1]
    objs[4].mlab_source.x = [vert[0] for vert in tran_v]
    objs[4].mlab_source.y = [vert[2] for vert in tran_v]
    objs[4].mlab_source.z = [vert[1] for vert in tran_v]
    objs[5].mlab_source.x = [vert[0] for vert in tran_v]
    objs[5].mlab_source.y = [vert[2] for vert in tran_v]
    objs[5].mlab_source.z = [vert[1] for vert in tran_v]
    objs[6].mlab_source.x = [vert[0] for vert in tran_v]
    objs[6].mlab_source.y = [vert[2] for vert in tran_v]
    objs[6].mlab_source.z = [vert[1] for vert in tran_v]
    objs[7].mlab_source.x = [vert[0] for vert in tran_v]
    objs[7].mlab_source.y = [vert[2] for vert in tran_v]
    objs[7].mlab_source.z = [vert[1] for vert in tran_v]

def put_queue(queue, mesh_v, meshes, points, PC_label, finished):
    points_FF = points[np.where(PC_label==0), :][0]
    points_LR = points[np.where(PC_label==1), :][0]
    points_RR = points[np.where(PC_label==2), :][0]
    points_SR = points[np.where(PC_label==3), :][0]
    queue[0].put(points_FF)
    queue[1].put(points_LR)
    queue[2].put(points_RR)
    queue[3].put(points_SR)
    queue[4].put(mesh_v)
    queue[5].put(finished)

#mesh_v: m*3
#mesh_f: k*3
def plot_mesh(mesh_v, mesh_f):
    mlab.triangular_mesh([vert[0] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         [vert[1] for vert in mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.show()

#mesh_v: m*3
#mesh_f: k*3
#mean: 1*3
#normal: 3*3
def plot_mesh_surfacenormal(mesh_v, mesh_f, mean, normal):
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.triangular_mesh([vert[0] for vert in mesh_v],
                         [vert[1] for vert in mesh_v],
                         [vert[2] for vert in mesh_v],
                         mesh_f,
                         opacity=0.5,
                         color=(0, 0, 1))
    origin = np.tile(mean, [3,1])
    mlab.quiver3d(origin[:, 0], origin[:, 1], origin[:, 2], normal[:,0], normal[:,1], normal[:,2], scale_factor=0.2)
    origin2 = np.asarray([[0.0,0.0,0.0],
                         [0.0,0.0,0.0],
                         [0.0,0.0,0.0]])
    cooridient = np.asarray([[1.0,0.0,0.0],
                             [0.0,1.0,0.0],
                             [0.0,0.0,1.0]])
    mlab.quiver3d(origin2[:, 0], origin2[:, 1], origin2[:, 2], cooridient[:,0], cooridient[:,1], cooridient[:,2], scale_factor=0.2)
    mlab.show()

#mesh_v1: m*3
#mesh_f1: k*3
#mesh_v2: m*3
#mesh_f2: k*3
def plot_meshes(mesh_v1, mesh_f1, mesh_v2, mesh_f2, back_index=None):
    if back_index != None:
        npoints = mesh_v1[back_index, :]
        mlab.points3d(npoints[:, 0], npoints[:, 2], npoints[:, 1], scale_factor=0.005, color=(1, 1, 1))
        npoints = mesh_v2[back_index, :]
        mlab.points3d(npoints[:, 0], npoints[:, 2], npoints[:, 1], scale_factor=0.005, color=(1, 0, 0))
    mlab.triangular_mesh([vert[0] for vert in mesh_v1],
                         [vert[2] for vert in mesh_v1],
                         [vert[1] for vert in mesh_v1],
                         mesh_f1,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.triangular_mesh([vert[0] for vert in mesh_v2],
                         [vert[2] for vert in mesh_v2],
                         [vert[1] for vert in mesh_v2],
                         mesh_f2,
                         opacity=0.5,
                         color=(1, 0, 0))
    mlab.show()


#mesh_v1: m*3
#mesh_f1: k*3
#mesh_v2: m*3
#mesh_f2: k*3
def plot_meshes_n_points(mesh_v1, mesh_f1, mesh_v2, mesh_f2, mesh_v3, mesh_f3, points, show=True):
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.016, color=(1, 1, 1))
    mlab.triangular_mesh([vert[0] for vert in mesh_v1],
                         [vert[1] for vert in mesh_v1],
                         [vert[2] for vert in mesh_v1],
                         mesh_f1,
                         opacity=0.5,
                         color=(0, 0, 1))
    s = mlab.triangular_mesh([vert[0] for vert in mesh_v2],
                         [vert[1] for vert in mesh_v2],
                         [vert[2] for vert in mesh_v2],
                         mesh_f2,
                         opacity=0.0,
                         color=(1, 0, 0))
    d = mlab.triangular_mesh([vert[0] for vert in mesh_v3],
                         [vert[1] for vert in mesh_v3],
                         [vert[2] for vert in mesh_v3],
                         mesh_f3,
                         opacity=0.5,
                         color=(0, 1, 0))
    if show:
        mlab.show()
    return s, d

#mesh_v1: m*3
#mesh_f1: k*3
#mesh_v2: m*3
#mesh_f2: k*3
#mesh_v2: m*3
#mesh_f2: k*3
def plot_3meshes(mesh_v1, mesh_f1, mesh_v2, mesh_f2, mesh_v3, mesh_f3):
    mlab.triangular_mesh([vert[0] for vert in mesh_v1],
                         [vert[2] for vert in mesh_v1],
                         [vert[1] for vert in mesh_v1],
                         mesh_f1,
                         opacity=0.5,
                         color=(0, 0, 1))
    mlab.triangular_mesh([vert[0] for vert in mesh_v2],
                         [vert[2] for vert in mesh_v2],
                         [vert[1] for vert in mesh_v2],
                         mesh_f2,
                         opacity=0.5,
                         color=(1, 0, 0))
    mlab.triangular_mesh([vert[0] for vert in mesh_v3],
                         [vert[2] for vert in mesh_v3],
                         [vert[1] for vert in mesh_v3],
                         mesh_f3,
                         opacity=0.5,
                         color=(0, 1, 0))
    mlab.show()

#mesh_v1: m*3
#mesh_f1: k*3
#mesh_t1: j*3
#node_force: m
def plot_mesh_force(tet_v, mesh_f1, v_t, node_force=None):
    if node_force == None:
        node_force = np.zeros(shape=tet_v.shape[0])
    mlab.figure(bgcolor=(1, 1, 1))
    '''mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         mesh_f1,
                         scalars=node_force,
                         opacity=0.3)'''
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         v_t[:,[0,1,2]],
                         scalars=node_force,
                         opacity=0.2)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         v_t[:,[0,1,3]],
                         scalars=node_force,
                         opacity=0.2)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         v_t[:,[0,2,3]],
                         scalars=node_force,
                         opacity=0.2)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         v_t[:,[1,2,3]],
                         scalars=node_force,
                         opacity=0.2)
    mlab.show()

def plot_tetr_mesh(tet_v, v_t, color, node_force=None):
    if node_force == None:
        node_force = np.zeros(shape=tet_v.shape[0])

    '''centers = tet_v[v_t[:, 0], :] + tet_v[v_t[:, 1], :] + tet_v[v_t[:, 2], :] + tet_v[v_t[:, 3], :]
    centers_mean = np.mean(centers, axis=0)
    v_t = v_t[np.where(centers[:,0]>centers_mean[0])[0], :]'''
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[0,1,2]],
                         color=color,)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[0,1,3]],
                         color=color,)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[0,2,3]],
                         color=color)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[1,2,3]],
                         color=color)

    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[0,1,2]],
                         scalars=node_force,
                         representation='wireframe',
                         color=(1, 0, 0),
                         opacity=0.2)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[0,1,3]],
                         scalars=node_force,
                         representation='wireframe',
                         color=(1, 0, 0),
                         opacity=0.2)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[0,2,3]],
                         scalars=node_force,
                         representation='wireframe',
                         color=(1, 0, 0),
                         opacity=0.2)
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         v_t[:,[1,2,3]],
                         scalars=node_force,
                         representation='wireframe',
                         color=(1, 0, 0),
                         opacity=0.2)



def plot_mesh_function(tet_v, mesh_f1, node_force):
    mlab.figure(bgcolor=(1, 1, 1))
    '''mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         mesh_f1,
                         scalars=node_force,
                         opacity=0.3)'''
    mlab.triangular_mesh([vert[0] for vert in tet_v],
                         [vert[2] for vert in tet_v],
                         [vert[1] for vert in tet_v],
                         mesh_f1,
                         scalars=node_force,
                         opacity=1)
    mlab.show()

def compute_n_plot_force(mesh, stiff, displacement):
    force = stiff.dot(np.reshape(displacement, [-1]))
    force = np.reshape(force, [-1, 3]).T
    node_force = np.sum(force * force, axis=0)
    print('displacement: ' + str(np.sum(displacement*displacement)))
    print('force:' + str(np.sum(node_force)))
    #plot_mesh_force(mesh.vertices, mesh.faces, mesh.voxels, node_force)

def plot_points_hightlight(points, points2):
    mlab.points3d(points[:, 0], points[:, 2], points[:, 1], scale_factor=0.02, color=(1, 1, 1))
    mlab.points3d(points2[:, 0], points2[:, 2], points2[:, 1], scale_factor=0.02, color=(1, 0, 0))
    mlab.show()