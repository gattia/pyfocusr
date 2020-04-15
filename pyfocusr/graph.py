import numpy as np
from scipy.linalg import eigh
import vtk


class Graph(object):
    def __init__(self,
                 vtk_mesh,
                 n_spectral_features=3,
                 norm_eig_vecs=True,
                 norm_points=True,
                 n_rand_samples=1000):

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
        self.laplacian_matrix = np.zeros_like(self.adjacency_matrix)
        self.G = None
        self.eig_vals = None
        self.eig_vecs = None
        self.n_spectral_features = n_spectral_features
        self.norm_eig_vecs = norm_eig_vecs
        self.eig_val_gap = None
        self.n_rand_samples = n_rand_samples
        self.rand_idxs = np.random.randint(0, self.n_points-1, self.n_rand_samples)

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
                distance = np.sqrt(np.sum(np.square(np.asarray(self.vtk_mesh.GetPoint(point_1)) -
                                                    np.asarray(self.vtk_mesh.GetPoint(point_2)))))
                self.adjacency_matrix[point_1, point_2] = 1. / distance

    def get_degree_matrix(self):
        for i in range(self.adjacency_matrix.shape[0]):
            self.degree_matrix[i, i] = np.sum(self.adjacency_matrix[i, :])

    def get_laplacian_matrix(self):
        if self.G is None:
            self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix

        elif self.G is not None:
            self.laplacian_matrix = np.matmul(np.linalg.inv(self.G), (self.degree_matrix - self.adjacency_matrix))

    def get_graph_spectrum(self):
        self.get_weighted_adjacency_matrix()
        self.get_degree_matrix()
        self.get_laplacian_matrix()

        self.eig_vals, self.eig_vecs = eigh(self.laplacian_matrix, eigvals=(1, self.n_spectral_features))
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



