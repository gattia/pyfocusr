import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse
# import vtk
from vtk.util.numpy_support import numpy_to_vtk
from .vtk_functions import *
from itkwidgets import Viewer
features_dictionary = {'curvature': get_min_max_curvature_values}


class Graph(object):
    def __init__(self,
                 vtk_mesh,
                 n_spectral_features=3,
                 norm_eig_vecs=True,
                 norm_points=True,
                 n_rand_samples=10000,
                 list_features_to_calc=[],
                 feature_weights=None):

        self.vtk_mesh = vtk_mesh
        self.n_points = vtk_mesh.GetNumberOfPoints()
        self.points = np.zeros((self.n_points, 3))
        for point_idx in range(self.n_points):
            self.points[point_idx, :] = self.vtk_mesh.GetPoint(point_idx)
        self.max_points_range = np.max(np.ptp(self.points, axis=0), axis=0)
        if norm_points is True:
            self.normalize_point_coordinates()
        else:
            self.norm_points = None

        self.adjacency_matrix = np.zeros((vtk_mesh.GetNumberOfPoints(),
                                          vtk_mesh.GetNumberOfPoints()))
        self.degree_matrix = np.zeros_like(self.adjacency_matrix)
        self.degree_matrix_inv = np.zeros_like(self.degree_matrix)
        self.laplacian_matrix = np.zeros_like(self.adjacency_matrix)
        self.G = None
        self.eig_vals = None
        self.eig_vecs = None
        self.n_spectral_features = n_spectral_features
        self.norm_eig_vecs = norm_eig_vecs
        self.eig_val_gap = None
        self.rand_idxs = self.get_list_rand_idxs(n_rand_samples)

        self.node_features = []
        for feature in list_features_to_calc:
            self.node_features += list(features_dictionary[feature](self.vtk_mesh))
        self.norm_node_features()  # normalize the node features to be in range 0-1, this makes everything else easier
        self.n_features = len(self.node_features)

        self.max_xyz_range_scaled_features = []
        if self.n_features > 0:
            for ftr_idx in range(len(self.node_features)):
                self.max_xyz_range_scaled_features.append(self.node_features[ftr_idx] * self.max_points_range)

        if feature_weights is None:
            self.feature_weights = np.eye(self.n_features)

    def norm_node_features(self):
        for idx in range(len(self.node_features)):
            self.node_features[idx] = (self.node_features[idx] - np.min(self.node_features[idx]))\
                                      / np.ptp(self.node_features[idx])

    def get_weighted_adjacency_matrix(self):
        '''
        Get/fill the adjacency matrix for the mesh vtk_mesh
        - Add options to enable adding the features
        :return:
        '''

        n_cells = self.vtk_mesh.GetNumberOfCells()
        for cell_idx in range(n_cells):
            cell = self.vtk_mesh.GetCell(cell_idx)
            for edge_idx in range(cell.GetNumberOfEdges()):
                edge = cell.GetEdge(edge_idx)
                point_1 = int(edge.GetPointId(0))
                point_2 = int(edge.GetPointId(1))

                X_pt1 = np.asarray(self.vtk_mesh.GetPoint(point_1))
                X_pt2 = np.asarray(self.vtk_mesh.GetPoint(point_2))

                if self.n_features > 0:
                    for ftr_idx in range(self.n_features):
                        # Append the "features" to the x/y/z position. Use features that have been scaled to be in
                        # the range of the max range axis of xyz.
                        X_pt1 = np.concatenate((X_pt1, self.max_xyz_range_scaled_features[ftr_idx][point_1, None]))
                        X_pt2 = np.concatenate((X_pt2, self.max_xyz_range_scaled_features[ftr_idx][point_2, None]))

                distance = np.sqrt(np.sum(np.square(X_pt1 -
                                                    X_pt2)))
                self.adjacency_matrix[point_1, point_2] = 1. / distance

    def get_G_matrix(self):
        if self.n_features > 0:
            self.G = np.zeros(self.n_points)
            for k in range(self.n_features):
                # Add up the normalized node _
                self.G += self.feature_weights[k, k] * np.exp(self.node_features[k])
            self.G = self.G / self.n_features
            self.G = np.diag(self.G)
            self.G = np.diag(self.degree_matrix_inv) * self.G
        elif self.n_features == 0:
            self.G = self.degree_matrix_inv

    def get_degree_matrix(self):
        for i in range(self.adjacency_matrix.shape[0]):
            self.degree_matrix[i, i] = np.sum(self.adjacency_matrix[i, :])
        self.degree_matrix_inv = np.diag((np.diag(self.degree_matrix) + 1e-8)**-1)

    def get_laplacian_matrix(self):
        # Ensure that G is defined.
        if self.G is None:
            self.G = self.degree_matrix_inv

        self.laplacian_matrix = self.G @ (self.degree_matrix - self.adjacency_matrix)

    def get_graph_spectrum(self):
        print('building adjacency matrix')
        self.get_weighted_adjacency_matrix()
        print('building degree matrix')
        self.get_degree_matrix()
        # self.get_G_matrix()
        print('starting to get laplacian matrix')
        self.get_laplacian_matrix()

        # sparse.csc_matrix was faster than sparse.csr_matrix on tests of 5k square matrix.
        # (359+/- 6.7 ms vs 379 +/- 20.2 ms  including 10 iterations per run and 7 runs).
        # providing sigma (a value to find eigenvalues near to) slows things down considerably.
        # providing `ncv` doesnt change things too much (maybe slower if anything).
        # The sparse versions are even faster than using eigh on a dense matrix.
        # Therefore, use sparse matrices for all circumstances.
        laplacian_sparse = sparse.csc_matrix(self.laplacian_matrix)
        print('beginning eigen decomposition')
        self.eig_vals, self.eig_vecs = eigs(laplacian_sparse,
                                            k=self.n_spectral_features+1,
                                            sigma=1e-10,
                                            which='LM')
        self.eig_vals = np.real(self.eig_vals[1: 1 + self.n_spectral_features])
        self.eig_vecs = np.real(self.eig_vecs[:, 1: 1 + self.n_spectral_features])
        if self.norm_eig_vecs is True:
            self.eig_vecs = (self.eig_vecs - np.min(self.eig_vecs, axis=0)) / np.ptp(self.eig_vecs, axis=0) - 0.5

    def get_eig_val_gap(self):
        self.eig_val_gap = np.mean(np.diff(self.eig_vals))

    def get_rand_eig_vecs(self):
        return self.eig_vecs[self.rand_idxs, :]

    def get_rand_normalized_points(self):
        return (self.points[self.rand_idxs, :] - np.min(self.points[self.rand_idxs, :], axis=0)) \
               / np.ptp(self.points[self.rand_idxs, :], axis=0)

    def normalize_point_coordinates(self):
        self.norm_points = (self.points - np.min(self.points, axis=0)) / self.max_points_range

    def view_mesh_existing_scalars(self):
        plotter = Viewer(geometries=[self.vtk_mesh]
                         )

        return plotter

    def view_mesh_eig_vec(self, eig_vec=0):
        tmp_mesh = self.vtk_mesh
        tmp_mesh.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(self.eig_vecs[:, eig_vec])))
        plotter = Viewer(geometries=[tmp_mesh]
                         )
        return plotter

    def view_mesh_features(self, feature_idx=0):
        tmp_mesh = self.vtk_mesh
        tmp_mesh.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray((self.node_features[feature_idx]))))
        plotter = Viewer(geometries=[tmp_mesh]
                         )
        return plotter

    def mean_filter_graph(self, values, iterations=300):
        """
        See below for copyright of this particular function:
        However, note that some changes have been made as the original was in Matlab, and included more options etc.

        Copyright (C) 2002, 2003 Leo Grady <lgrady@cns.bu.edu>
        Computer Vision and Computational Neuroscience Lab
        Department of Cognitive and Neural Systems
        Boston University
        Boston, MA  02215

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to the Free Software
        Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

        :param values:
        :param iterations:
        :return:
        """
        D_inv = np.diag(1./(1+np.sum(self.adjacency_matrix, axis=0)))
        out_values = values
        average_mat = D_inv @ (self.adjacency_matrix + np.eye(self.adjacency_matrix.shape[0]))
        for iteration in range(iterations):
            out_values = average_mat @ out_values
        return out_values

    def get_list_rand_idxs(self, n_rand_samples, replace=False, force_randomization=False):
        """
        Return idxs of random samples
        - By default do not use replacement (each sample should only be able to be taken one)
        - If n_rand_samples is more than the number of points, should just return idxs to all points.
        :param force_randomization:
        :param n_rand_samples:
        :param replace:
        :return:
        """
        if n_rand_samples > self.n_points:
            list_points = np.arange(self.n_points)
            if force_randomization is True:
                np.shuffle(list_points)
            return list_points

        return np.random.choice(self.n_points, size=n_rand_samples, replace=replace)


