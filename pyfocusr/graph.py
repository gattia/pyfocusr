import numpy as np
from itkwidgets import Viewer
from scipy import sparse
from scipy.sparse.linalg import eigs

# import vtk
from vtk.util.numpy_support import numpy_to_vtk

from .vtk_functions import *

features_dictionary = {
    "curvature": get_min_max_curvature_values,
    "min_curvature": get_min_curvature,
    "max_curvature": get_max_curvature,
}


class Graph(object):
    def __init__(
        self,
        vtk_mesh,
        n_spectral_features=3,
        norm_eig_vecs=True,
        n_rand_samples=10000,
        list_features_to_calc=[],
        list_features_to_get_from_mesh=[],
        feature_weights=None,
        include_features_in_adj_matrix=False,
        include_features_in_G_matrix=False,
        G_matrix_p_function="exp",
        norm_node_features_std=True,
        norm_node_features_cap_std=3,
        norm_node_features_0_1=True,
    ):

        # Inputs
        self.vtk_mesh = vtk_mesh  # store mesh
        self.n_spectral_features = n_spectral_features  # number of spectral features to extract.
        self.norm_eig_vecs = norm_eig_vecs  # Bool - to normalize eigvecs or not.

        self.feature_weights = feature_weights  # Prep feature weights
        if feature_weights is None:
            self.feature_weights = np.eye(self.n_extra_features)
        else:
            self.feature_weights = feature_weights
        self.include_features_in_adj_matrix = (
            include_features_in_adj_matrix  # Bool, features in adj?
        )
        self.include_features_in_G_matrix = (
            include_features_in_G_matrix  # Bool, include features in G
        )
        self.G_matrix_p_function = G_matrix_p_function
        # How to normmalize extra features.
        self.norm_node_features_std = norm_node_features_std
        self.norm_node_features_cap_std = norm_node_features_cap_std
        self.norm_node_features_0_1 = norm_node_features_0_1

        # Mesh/points characteristics
        self.n_points = vtk_mesh.GetNumberOfPoints()  # store number points in mesh.
        # Iterate over the points saving their 3d location.
        self.points = np.zeros((self.n_points, 3))
        for point_idx in range(self.n_points):
            self.points[point_idx, :] = self.vtk_mesh.GetPoint(point_idx)
        self.pts_scale_range = np.ptp(self.points, axis=0)  # range points in each axis
        self.max_pts_scale_range = np.max(self.pts_scale_range)  # max range points
        self.mean_pts_scale_range = np.mean(self.pts_scale_range)  # mean range points (per axis)
        # create normalized version of point coordinates.
        self.normed_points = (self.points - np.min(self.points, axis=0)) / self.mean_pts_scale_range

        # Assign matrices that will be used for laplacian and eigen decomposition.
        self.adjacency_matrix = sparse.lil_matrix(
            (vtk_mesh.GetNumberOfPoints(), vtk_mesh.GetNumberOfPoints())
        )
        self.degree_matrix = None
        self.degree_matrix_inv = None
        self.laplacian_matrix = None
        self.G = None

        # Eigen values that will be calculated & their characteristics.
        self.eig_vals = None
        self.eig_vecs = None
        self.eig_val_gap = None
        self.rand_idxs = self.get_list_rand_idxs(n_rand_samples)

        # Calculate node features & store in self.node_features for use.
        self.node_features = []
        for feature in list_features_to_calc:
            self.node_features += list(features_dictionary[feature](self.vtk_mesh))
        for feature in list_features_to_get_from_mesh:
            n = vtk_mesh.GetPointData().GetNumberOfArrays()
            for idx in range(n):
                if vtk_mesh.GetPointData().GetArray(idx).GetName() == feature:
                    break
                elif idx == n - 1:
                    print("NO SCALARS WITH SPECIFIED NAME")
                    idx = np.nan
                    break
                else:
                    pass

            self.node_features += list(
                [
                    vtk_to_numpy(vtk_mesh.GetPointData().GetArray(idx)),
                ]
            )

        # normalize the node features w/ options for how it is normalized.
        self.norm_node_features(
            norm_using_std=self.norm_node_features_std,
            norm_range_0_to_1=self.norm_node_features_0_1,
            cap_std=self.norm_node_features_cap_std,
        )
        self.n_extra_features = len(self.node_features)  # number of extra features used.
        # Get version of extra features that are scaled up to the
        self.mean_xyz_range_scaled_features = []
        if self.n_extra_features > 0:
            for ftr_idx in range(len(self.node_features)):
                self.mean_xyz_range_scaled_features.append(
                    self.node_features[ftr_idx] * self.mean_pts_scale_range
                )

    def norm_node_features(self, norm_using_std=True, norm_range_0_to_1=True, cap_std=3):
        """
        Need multiple methods of normalizing the node_features.

        :param cap_std:
        :param norm_range_0_to_1:
        :param norm_using_std:
        :return:
        """
        for idx in range(len(self.node_features)):
            if norm_using_std is True:
                self.node_features[idx] = (
                    self.node_features[idx] - np.mean(self.node_features[idx])
                ) / np.std(self.node_features[idx])
                if cap_std is not False:
                    self.node_features[idx][self.node_features[idx] > cap_std] = cap_std
                    self.node_features[idx][self.node_features[idx] < -cap_std] = -cap_std

            if norm_range_0_to_1 is True:
                self.node_features[idx] = (
                    self.node_features[idx] - np.min(self.node_features[idx])
                ) / np.ptp(self.node_features[idx])

    """
    Functions to create matrices needed for laplacian and eigen decomposition. 
    """

    def get_weighted_adjacency_matrix(self):
        """
        Get/fill the adjacency matrix for the mesh vtk_mesh
        - Add options to enable adding the features
        :return:
        """

        n_cells = self.vtk_mesh.GetNumberOfCells()
        for cell_idx in range(n_cells):
            cell = self.vtk_mesh.GetCell(cell_idx)
            for edge_idx in range(cell.GetNumberOfEdges()):
                edge = cell.GetEdge(edge_idx)
                point_1 = int(edge.GetPointId(0))
                point_2 = int(edge.GetPointId(1))

                X_pt1 = np.asarray(self.vtk_mesh.GetPoint(point_1))
                X_pt2 = np.asarray(self.vtk_mesh.GetPoint(point_2))

                if (self.n_extra_features > 0) & (self.include_features_in_adj_matrix is True):
                    for ftr_idx in range(self.n_extra_features):
                        # Append the "features" to the x/y/z position. Use features that have been scaled to be in
                        # the range of the max range axis of xyz.
                        X_pt1 = np.concatenate(
                            (X_pt1, self.mean_xyz_range_scaled_features[ftr_idx][point_1, None])
                        )
                        X_pt2 = np.concatenate(
                            (X_pt2, self.mean_xyz_range_scaled_features[ftr_idx][point_2, None])
                        )

                distance = np.sqrt(np.sum(np.square(X_pt1 - X_pt2)))
                self.adjacency_matrix[point_1, point_2] = 1.0 / distance

    def get_G_matrix(self, p_function="exp"):
        """
        Get G matrix for creating laplacian laplacian = G * (D-W)
        p_function options include:
            - exp
            - log
            - square
            -otherwise just make sure it is 0 or higher.
        :param p_function:
        :return:
        """
        if (self.n_extra_features > 0) & (self.include_features_in_G_matrix is True):
            self.G = np.zeros(self.n_points)
            for k in range(self.n_extra_features):
                # Add up the normalized node _
                if p_function == "exp":
                    G = np.exp(self.node_features[k])
                elif p_function == "log":
                    # use log function. Ensure that all values are above zero (make it 1 and up).
                    G = np.log(self.node_features[k] - np.min(self.node_features[k]) + 1)
                elif p_function == "square":
                    G = self.node_features[k] ** 2
                else:
                    # Otherwise, just ensure features are 0 and higher.
                    G = self.node_features[k] - np.min(self.node_features[k])
                # Scale features to be in range of degree_matrix. Then, multople by the feature weighting.
                G_scaling = self.feature_weights[k, k] * np.ptp(self.degree_matrix) / np.ptp(G)
                self.G += G * G_scaling  # Add scaled feature values to to G matrix.
            self.G = self.G / self.n_extra_features  # Get average self.G across features.
            self.G = sparse.diags(self.G)
            self.G = self.G.multiply(self.degree_matrix_inv.diagonal())
            # self.G = self.degree_matrix_inv @ self.G

        else:
            self.G = self.degree_matrix_inv

    def get_degree_matrix(self):
        self.degree_matrix = np.asarray(self.adjacency_matrix.sum(axis=1))
        self.degree_matrix = sparse.diags(self.degree_matrix[:, 0])
        self.degree_matrix_inv = sparse.diags((self.degree_matrix.diagonal() + 1e-8) ** -1)

    def get_laplacian_matrix(self):
        # Ensure that G is defined.
        if self.G is None:
            self.G = self.degree_matrix_inv
        laplacian = self.degree_matrix - self.adjacency_matrix
        self.laplacian_matrix = self.G @ laplacian

    def get_graph_spectrum(self):
        self.get_weighted_adjacency_matrix()
        self.get_degree_matrix()
        self.get_G_matrix(p_function=self.G_matrix_p_function)
        self.get_laplacian_matrix()

        # sparse.csc_matrix was faster than sparse.csr_matrix on tests of 5k square matrix.
        # (359+/- 6.7 ms vs 379 +/- 20.2 ms  including 10 iterations per run and 7 runs).
        # providing sigma (a value to find eigenvalues near to) slows things down considerably.
        # providing `ncv` doesnt change things too much (maybe slower if anything).
        # The sparse versions are even faster than using eigh on a dense matrix.
        # Therefore, use sparse matrices for all circumstances.
        # laplacian_sparse = sparse.csc_matrix(self.laplacian_matrix)
        print("Beginning Eigen Decomposition")

        self.eig_vals, self.eig_vecs = recursive_eig(
            self.laplacian_matrix,
            k=self.n_spectral_features + 1,
            n_k_needed=self.n_spectral_features,
            k_buffer=1,
        )

        print("All final eigenvalues are: \n{}".format(self.eig_vals))
        print("-" * 72)
        print("Final eigenvalues of interest are: \n{}".format(self.eig_vals))

        if self.norm_eig_vecs is True:
            self.eig_vecs = (self.eig_vecs - np.min(self.eig_vecs, axis=0)) / np.ptp(
                self.eig_vecs, axis=0
            ) - 0.5

    """
    Get sub samples/measurements from/of eigenvectors or characteristics about them. 
    """

    def get_eig_val_gap(self):
        self.eig_val_gap = np.mean(np.diff(self.eig_vals))

    def get_rand_eig_vecs(self):
        return self.eig_vecs[self.rand_idxs, :]

    def get_rand_normalized_points(self):
        return (
            self.points[self.rand_idxs, :] - np.min(self.points[self.rand_idxs, :], axis=0)
        ) / np.ptp(self.points[self.rand_idxs, :], axis=0)

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

    """
    View meshes/points/results. 
    """

    def view_mesh_existing_scalars(self):
        plotter = Viewer(geometries=[self.vtk_mesh])
        return plotter

    def view_mesh_eig_vec(self, eig_vec=0):
        tmp_mesh = vtk_deep_copy(self.vtk_mesh)
        tmp_mesh.GetPointData().SetScalars(
            numpy_to_vtk(np.ascontiguousarray(self.eig_vecs[:, eig_vec]))
        )
        plotter = Viewer(geometries=[tmp_mesh])
        return plotter

    def view_mesh_features(self, feature_idx=0):
        tmp_mesh = vtk_deep_copy(self.vtk_mesh)
        tmp_mesh.GetPointData().SetScalars(
            numpy_to_vtk(np.ascontiguousarray((self.node_features[feature_idx])))
        )
        plotter = Viewer(geometries=[tmp_mesh])
        return plotter

    """
    Filter graph f(x)s 
    """

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
        D_inv = sparse.diags(1.0 / (1 + np.asarray(self.adjacency_matrix.sum(axis=1))[:, 0]))
        out_values = values
        average_mat = D_inv @ (self.adjacency_matrix + sparse.eye(self.adjacency_matrix.shape[0]))
        for iteration in range(iterations):
            out_values = average_mat @ out_values
        return out_values


def recursive_eig(matrix, k, n_k_needed, k_buffer=1, sigma=1e-10, which="LM"):
    """
    Recursive function to iteratively get eigs until have enough to get fiedler + n_k_needed @ minimum.
    If one final
    :param matrix:
    :param k:
    :param n_k_needed:
    :param k_buffer:
    :param sigma:
    :param which:
    :return:
    """
    MIN_EIG_VAL = 1e-10

    print("Starting!")
    eig_vals, eig_vecs = eigs(matrix, k=k, sigma=sigma, which=which, ncv=4 * k)

    n_good_eigen_vals = sum(eig_vals > MIN_EIG_VAL)

    if n_good_eigen_vals < n_k_needed:
        print("Not enough eigenvalues found, trying again with more eigenvalues!")
        k += k_buffer + n_k_needed
        eig_vals, eig_vecs = recursive_eig(matrix, k, n_k_needed, k_buffer, sigma, which)

    eig_keep = np.where(eig_vals > MIN_EIG_VAL)[0]

    eig_vals = eig_vals[eig_keep]
    eig_vecs = eig_vecs[:, eig_keep]

    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    return eig_vals, eig_vecs
