import pymesh
from pymesh import Material
import trimesh
from mayavi import mlab
from util_package import plot
from util_package import util
import numpy as np
import copy
from scipy.interpolate import RegularGridInterpolator
from util_package.solver import solve_one_quadratic_new
from scipy.spatial import Voronoi, Delaunay


class Tetr_Mesh:
    def __init__(self, tumer_core, whole_tumer, whole_brain):
        # surface_mesh is the surface that represents this volume, brain without tumor core
        # edma_surface is the surface that represent the whole tumor, tumor core + edema
        # vertive_label

        #
        self.edema_surface, _ = pymesh.collapse_short_edges(whole_tumer, abs_threshold=3.0)
        mesh = pymesh.merge_meshes([pymesh.form_mesh(tumer_core.vertices, tumer_core.faces), pymesh.form_mesh(whole_brain.vertices, whole_brain.faces)])
        self.tetr_mesh = pymesh.tetrahedralize(mesh, cell_size=10, engine='cgal_no_features')
        #self.form_tetr_mesh(tumer_core, whole_brain)
        self.output_tetr_mesh = self.tetr_mesh

        material = Material.create_isotropic(3, 1.0, young=694, poisson=0.4) # parameters come from "Simulation of Brain Tumors in MR Images for Evaluation of Segmentation Efficacy"
        assembler = pymesh.Assembler(self.tetr_mesh, material=material)
        self.stiffness = assembler.assemble("stiffness")
        self.laplacian = assembler.assemble("laplacian")

        self.tumer_v_index, self.brain_v_index = self.extract_surface_vertices()
        self.edema_voxels, self.edema_voxel_labels = self.extract_edema_voxels()


    def debug_solve(self, plot=True):
        if plot:
            self.plot()
        vertices = self.get_tumor_vertices()
        displacement = (np.tile(np.mean(vertices, axis=0), (vertices.shape[0], 1)) - vertices)
        #displacement = self.method1()
        self.solve_for_displacement(displacement)
        if plot:
            self.plot()
        #plot.plot_tetr_mesh(self.tetr_mesh.vertices, self.tetr_mesh.voxels)

    def form_tetr_mesh(self, tumer_core, whole_brain):
        self.tumor_surface, _ = pymesh.collapse_short_edges(pymesh.form_mesh(tumer_core.vertices, tumer_core.faces), abs_threshold=2.0)
        self.tumor_surface, _ = pymesh.remove_isolated_vertices(self.tumor_surface)
        tumer_core = trimesh.Trimesh(self.tumor_surface.vertices, self.tumor_surface.faces)
        tetr_whole = pymesh.tetrahedralize(pymesh.form_mesh(whole_brain.vertices, whole_brain.faces), cell_size=6, engine='cgal_no_features')
        vertices_inside_tumor_index = tumer_core.contains(tetr_whole.vertices)
        vertices_outside_tumor_index = 1 * np.logical_not(vertices_inside_tumor_index)
        vertices_inside_tumor_index = 1 * vertices_inside_tumor_index
        cross_tumer_surface_voxels = vertices_inside_tumor_index[tetr_whole.voxels[:, 0]] \
                                     + vertices_inside_tumor_index[tetr_whole.voxels[:, 1]] \
                                     + vertices_inside_tumor_index[tetr_whole.voxels[:, 2]] \
                                     + vertices_inside_tumor_index[tetr_whole.voxels[:, 3]]

        sq_dis, _, _ = pymesh.distance_to_mesh(self.tumor_surface, tetr_whole.vertices)
        distance = np.sqrt(sq_dis)
        vertices_close_to_tumor = 1 * (distance < 3)
        voxels_close_to_surface = vertices_close_to_tumor[tetr_whole.voxels[:, 0]] \
                                  + vertices_close_to_tumor[tetr_whole.voxels[:, 1]] \
                                  + vertices_close_to_tumor[tetr_whole.voxels[:, 2]] \
                                  + vertices_close_to_tumor[tetr_whole.voxels[:, 3]]

        voxels_around_surface = tetr_whole.voxels[np.where(
            np.logical_or(np.logical_or(cross_tumer_surface_voxels==2, cross_tumer_surface_voxels==3), cross_tumer_surface_voxels==1)
        )]
        vertices_around_surface = tetr_whole.vertices[np.unique(voxels_around_surface), :]

        self.brain_surface = pymesh.form_mesh(tetr_whole.vertices, tetr_whole.faces)
        self.brain_surface, _ = pymesh.remove_isolated_vertices(self.brain_surface)

        all_points = np.concatenate([vertices_around_surface, self.tumor_surface.vertices])
        _, all_points_index_org = util.closest(all_points, tetr_whole.vertices)
        all_points_index_org[vertices_around_surface.shape[0]:] = tetr_whole.vertices.shape[0] + np.asarray(range(self.tumor_surface.vertices.shape[0]))
        num_of_vertices = vertices_around_surface.shape[0]
        Delaunay_tri = Delaunay(all_points)
        candicates_voxels = Delaunay_tri.simplices
        candicates_voxels_centers = np.sum(all_points[candicates_voxels, :], axis=1) / 4

        valid_voxels = np.logical_and(
            np.logical_not(tumer_core.contains(candicates_voxels_centers)),
            np.logical_or(np.logical_or(candicates_voxels[:,0]>=num_of_vertices, candicates_voxels[:,1]>=num_of_vertices),
                          np.logical_or(candicates_voxels[:,2]>=num_of_vertices, candicates_voxels[:,3]>=num_of_vertices))
        )

        '''mesh1 = pymesh.form_mesh(tetr_whole.vertices, tetr_whole.faces,
                                 tetr_whole.voxels[np.where(cross_tumer_surface_voxels == 0)])
        mesh2 = pymesh.form_mesh(all_points, self.tumor_surface.faces + vertices_around_surface.shape[0], candicates_voxels[np.where(valid_voxels)[0], :])
        self.tetr_mesh = pymesh.merge_meshes([mesh1, mesh2])'''

        self.tetr_mesh = pymesh.form_mesh(vertices=np.concatenate([tetr_whole.vertices, self.tumor_surface.vertices]),
                                          faces=np.concatenate([tetr_whole.faces,
                                                                self.tumor_surface.faces + tetr_whole.vertices.shape[0]]),
                                          voxels=np.concatenate([tetr_whole.voxels[np.where(cross_tumer_surface_voxels == 0)],
                                                                 all_points_index_org[candicates_voxels[np.where(valid_voxels)[0], :]]]))
        #self.tetr_mesh, _ = pymesh.remove_isolated_vertices(self.tetr_mesh)
        #self.tetr_mesh, _ = pymesh.collapse_short_edges(self.tetr_mesh, abs_threshold=1.0e-10)
        #self.tetr_mesh, _ = pymesh.remove_duplicated_vertices(self.tetr_mesh)

        #self.output_tetr_mesh = self.tetr_mesh
        #self.plot(np.where(cross_tumer_surface_voxels ==4), show=False)
        #mlab.points3d(vertices_around_surface[:, 0], vertices_around_surface[:, 1], vertices_around_surface[:, 2], scale_factor=0.1)
        #mlab.show()



    def extract_edema_voxels(self):
        voxel_centers = np.sum(self.tetr_mesh.vertices[self.tetr_mesh.voxels, :], axis=1) / 4
        mesh = trimesh.Trimesh(self.edema_surface.vertices, self.edema_surface.faces)
        is_edema = mesh.contains(voxel_centers)
        return self.tetr_mesh.voxels[np.where(is_edema==True)[0], :], is_edema

    def extract_surface_vertices(self):
        meshes = pymesh.separate_mesh(pymesh.form_mesh(self.tetr_mesh.vertices, self.tetr_mesh.faces))
        if np.abs(meshes[0].volume) > np.abs(meshes[1].volume):
            self.brain_surface = meshes[0]
            self.tumor_surface = meshes[1]
        else:
            self.brain_surface = meshes[1]
            self.tumor_surface = meshes[0]
        distance, closest_brain = util.closest(self.brain_surface.vertices, self.tetr_mesh.vertices)
        distance, closest_tumor = util.closest(self.tumor_surface.vertices, self.tetr_mesh.vertices)
        return closest_tumor, closest_brain

    def solve_for_displacement(self, tumor_surface_displacement):
        displacement_boundary = np.zeros_like(self.tetr_mesh.vertices)
        displacement_boundary[self.tumer_v_index, :] = tumor_surface_displacement
        self.output_tetr_mesh = solve_one_quadratic_new(self.tetr_mesh,
                                                        self.stiffness, self.laplacian,
                                                        np.concatenate([self.tumer_v_index, self.brain_v_index]), displacement_boundary)
        self.output_displacement = self.output_tetr_mesh.vertices - self.tetr_mesh.vertices

    def interpolate_displacement(self, org_data):
        index = np.where(org_data != 0)
        voxels_interpolate = np.concatenate([np.expand_dims(index[0], axis=1),
                                             np.expand_dims(index[1], axis=1),
                                             np.expand_dims(index[2], axis=1)], axis=1)
        #_, closest = util.closest(voxels_interpolate, self.tetr_mesh.vertices)
        voxels_moved = util.tetra_interpolation_full(voxels_interpolate, self.output_tetr_mesh.vertices, self.output_displacement)
        #voxels_moved = util.tetra_interpolation_delaunay(voxels_interpolate, self.output_tetr_mesh.vertices, self.output_displacement)
        voxels_moved = voxels_interpolate - voxels_moved[0]

        shape = org_data.shape
        x = np.linspace(0, shape[0], shape[0])
        y = np.linspace(0, shape[1], shape[1])
        z = np.linspace(0, shape[2], shape[2])
        fn = RegularGridInterpolator((x, y, z), org_data)
        value = fn(voxels_moved)

        out_data = np.zeros_like(org_data)
        out_data[index] = value
        return out_data


    #################################################################################################################
    #################################################################################################################
    def plot(self, valid_voxels_2=None, show=True):
        mlab.figure(bgcolor=(1, 1, 1))
        mlab.view(azimuth=-10, elevation=100, roll=80)
        tumor_center = np.mean(self.tumor_surface.vertices, axis=0)
        voxel_centers = np.sum(self.output_tetr_mesh.vertices[self.output_tetr_mesh.voxels,:], axis=1) / 4
        #valid_voxels_1 = self.tetr_mesh.voxels[np.where(np.logical_or(np.logical_or((voxel_centers[:,0] > tumor_center[0]), (voxel_centers[:,1] > tumor_center[1])),
        #                                                              (voxel_centers[:, 2] > tumor_center[2])
        #                                                              ))[0], :]
        valid_voxels_1 = self.output_tetr_mesh.voxels[np.where((voxel_centers[:,0] > tumor_center[0]))[0], :]
        plot.plot_tetr_mesh(self.output_tetr_mesh.vertices, valid_voxels_1, (0.5, 0.5, 0.5))

        if valid_voxels_2 == None:
            valid_voxels_2 = np.where(np.logical_and((voxel_centers[:,0] > tumor_center[0]), self.edema_voxel_labels))[0]
        plot.plot_tetr_mesh(self.output_tetr_mesh.vertices, self.output_tetr_mesh.voxels[valid_voxels_2], (1.0, 1.0, 0.0))
        mlab.triangular_mesh([vert[0] for vert in self.brain_surface.vertices],
                             [vert[1] for vert in self.brain_surface.vertices],
                             [vert[2] for vert in self.brain_surface.vertices],
                             self.brain_surface.faces,
                             opacity=0.2,
                             color=(0, 0, 1))
        mlab.triangular_mesh([vert[0] for vert in self.tumor_surface.vertices],
                             [vert[1] for vert in self.tumor_surface.vertices],
                             [vert[2] for vert in self.tumor_surface.vertices],
                             self.tumor_surface.faces,
                             opacity=0.4,
                             color=(0, 1, 0))
        if show:
            mlab.show()
    #################################################################################################################
    #################################################################################################################

    def get_tumor_mesh(self):
        return self.tumor_surface

    def get_tumor_vertices(self):
        return self.tumor_surface.vertices

    def get_brain_mesh(self):
        return self.brain_surface

    def get_brain_vertices(self):
        return self.brain_surface.vertices


