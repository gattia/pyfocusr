import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .graph import Graph
import cycpd
from itkwidgets import Viewer
from matplotlib import colors
from .vtk_functions import *
from .main import *
from .eigsort import eigsort
import time


class Focusr(object):
    def __init__(self,
                 vtk_mesh_target,
                 vtk_mesh_source,
                 icp_register_first=True,               # bool, should register meshes together first
                 icp_registration_mode='rigid',         # str - should icp reg be rigid or similarity (rigid + scale)
                 icp_reg_target_to_source=False,        # bool of what should be registered to what in ICP.
                 n_spectral_features=3,                 #
                 n_extra_spectral=3,                    #
                 target_eigenmap_as_reference=True,     # bool, should use target eigenmap as reference for spectral reg
                                                        # This is helpful for registering a "template" mesh (which is the source)
                                                        # to another mesh (which is the target), commonly performed in
                                                        # statistical shape models (SSM).
                 norm_physical_and_spectral=True,       #
                 n_coords_spectral_ordering=5000,       #
                 n_coords_spectral_registration=5000,   #
                 rigid_before_non_rigid_reg=True,       #
                 rigid_reg_max_iterations=100,          #
                 rigid_tolerance=1e-8,                  #
                 non_rigid_max_iterations=1000,         #
                 non_rigid_tolerance=1e-8,              #
                 non_rigid_alpha=0.5,                   #
                 non_rigid_beta=3.0,                    #
                 non_rigid_n_eigens=100,                #
                 include_points_as_features=False,      #
                 get_weighted_spectral_coords=True,     #
                 graph_smoothing_iterations=300,        #
                 feature_smoothing_iterations=40,       #
                 smooth_correspondences=True,           #
                 return_average_final_points=True,      # should we create diffused/weighted new points
                 return_nearest_final_points=True,      # should we create nearest neighbor new points.
                 return_transformed_mesh=True,          #
                 projection_smooth_iterations=40,       #
                 feature_weights=None,                  #
                 initial_correspondence_type='kd',      # 'kd' or 'hungarian'
                 final_correspondence_type='kd',        #
                 list_features_to_calc=['curvature'],   # include as input of graph_source & graph_target
                 list_features_to_get_from_mesh=[],     # 'thickness (mm)' - get info from mesh surface for reg
                 use_features_as_coords=False,          #
                 use_features_in_graph=False,           #
                 include_features_in_adj_matrix=False,  # include as input of graph_source & graph_target
                 G_matrix_p_function='exp',             # Param for feature processing before laplacian creation
                 norm_node_features_std=True,           # Param for feature processing before laplacian creation
                 norm_node_features_cap_std=3,          # Param for feature processing before laplacian creation
                 norm_node_features_0_1=True,           # Param for feature processing before laplacian creation
                 verbose=False                          # bool - setting whether should print extraneous details
                 ):
        self.verbose = verbose
        print('Starting Focusr')
        # Inputs
        #   Spectral coordinates based inputs/parameters
        self.n_spectral_features = n_spectral_features
        self.n_extra_spectral = n_extra_spectral
        self.n_total_spectral_features = self.n_spectral_features + self.n_extra_spectral
        self.target_eigenmap_as_reference = target_eigenmap_as_reference

        #   Normalization & what included in registration
        self.norm_physical_and_spectral = norm_physical_and_spectral  # Bool, norm spect&xyz. Otherwise, spect to xyz.
        self.include_points_as_features = include_points_as_features  # include xyz coords in registration
        self.get_weighted_spectral_coords = get_weighted_spectral_coords
        self.feature_smoothing_iterations = feature_smoothing_iterations  # number smooth iterations to extra features
        #   Registration parameters (general)
        self.n_coords_spectral_registration = n_coords_spectral_registration  # max n points for registration
        #       Rigid reg params
        self.rigid_before_non_rigid_reg = rigid_before_non_rigid_reg
        self.rigid_reg_max_iterations = rigid_reg_max_iterations
        self.rigid_tolerance = rigid_tolerance
        #       Deformable reg params
        self.non_rigid_max_iterations = non_rigid_max_iterations
        self.non_rigid_tolerance = non_rigid_tolerance
        self.non_rigid_alpha = non_rigid_alpha
        self.non_rigid_beta = non_rigid_beta
        self.non_rigid_n_eigens = non_rigid_n_eigens
        #   Correspondence selection parameters
        self.initial_correspondence_type = initial_correspondence_type
        self.smooth_correspondences = smooth_correspondences  # Bool - smooth values to improve diffeomorphism?
        self.return_average_final_points = return_average_final_points  # make weighted avg final xyz position?
        self.return_nearest_final_points = return_nearest_final_points  # make nearest neighbour final xyz position?
        self.graph_smoothing_iterations = graph_smoothing_iterations
        self.projection_smooth_iterations = projection_smooth_iterations  # n iterations projection smoothing
        self.final_correspondence_type = final_correspondence_type  # 'kd' or 'hungarian' correspondence
        self.return_transformed_mesh = return_transformed_mesh  # bool to tell if we should create new mesh.

        print('Starting ICP')
        # Prepare Meshes / Graphs
        #   Rigidly register target to source before beginning.
        #   This ensures they are in same space for all steps.
        if icp_register_first is True:
            if icp_reg_target_to_source is True:
                if icp_registration_mode == 'rigid':
                    icp = icp_transform(target=vtk_mesh_source, source=vtk_mesh_target, transform_mode='rigid')
                elif icp_registration_mode == 'similarity':
                    icp = icp_transform(target=vtk_mesh_source, source=vtk_mesh_target, transform_mode='similarity')
                vtk_mesh_target = apply_transform(source=vtk_mesh_target, transform=icp)
            elif icp_reg_target_to_source is False:
                if icp_registration_mode == 'rigid':
                    icp = icp_transform(target=vtk_mesh_target, source=vtk_mesh_source, transform_mode='rigid')
                elif icp_registration_mode == 'similarity':
                    icp = icp_transform(target=vtk_mesh_target, source=vtk_mesh_source, transform_mode='similarity')
                vtk_mesh_source = apply_transform(source=vtk_mesh_source, transform=icp)
        print('Starting to build first graph')
        # Build target graph
        self.graph_target = Graph(vtk_mesh_target,
                                  n_spectral_features=self.n_total_spectral_features,
                                  n_rand_samples=n_coords_spectral_ordering,
                                  list_features_to_calc=list_features_to_calc,
                                  list_features_to_get_from_mesh=list_features_to_get_from_mesh,
                                  feature_weights=feature_weights,
                                  include_features_in_G_matrix=use_features_in_graph,
                                  include_features_in_adj_matrix=include_features_in_adj_matrix,
                                  G_matrix_p_function=G_matrix_p_function,
                                  norm_node_features_std=norm_node_features_std,
                                  norm_node_features_cap_std=norm_node_features_cap_std,
                                  norm_node_features_0_1=norm_node_features_0_1
                                  )
        print('Loaded Mesh 1')
        # Build target spectrum
        self.graph_target.get_graph_spectrum()
        print('Computed spectrum 1')
        # Build source graph
        self.graph_source = Graph(vtk_mesh_source,
                                  n_spectral_features=self.n_total_spectral_features,
                                  n_rand_samples=n_coords_spectral_ordering,
                                  list_features_to_calc=list_features_to_calc,
                                  list_features_to_get_from_mesh=list_features_to_get_from_mesh,
                                  feature_weights=feature_weights,
                                  include_features_in_G_matrix=use_features_in_graph,
                                  include_features_in_adj_matrix=include_features_in_adj_matrix,
                                  G_matrix_p_function=G_matrix_p_function,
                                  norm_node_features_std=norm_node_features_std,
                                  norm_node_features_cap_std=norm_node_features_cap_std,
                                  norm_node_features_0_1=norm_node_features_0_1,
                                  )
        print('Loaded Mesh 2')
        # Build source spectrum
        self.graph_source.get_graph_spectrum()
        print('Computed spectrum 2')

        # Define / specify parameters to be used.
        # Spectral alignment related
        self.Q = None
        self.spec_weights = None
        self.source_spectral_coords = None  # Could pre-allocate using np.zeros()
        self.target_spectral_coords = None  # Could pre-allocate using np.zeros()

        # Extra features (curvature etc.)
        self.source_extra_features = None  # Extra features used for mapping
        self.target_extra_features = None
        self.use_features_as_coords = use_features_as_coords

        # Saved versions of spectral coords during registration/processing for post-analysis/viewing
        self.source_spectral_coords_after_rigid = None
        self.source_spectral_coords_b4_reg = None

        # saved registration parameters - not currently used for anything. Could be used to transform points after
        # the fact.
        self.rigid_params = None
        self.non_rigid_params = None

        # Results / Correspondences:

        self.smoothed_target_coords = None       # smoothed coordinates of target - used for final correspondences.
        self.source_projected_on_target = None   # source values projected on target graph for finding final correspond
        self.weighted_avg_transformed_mesh = None  # source mesh transformed to target w/ weighted avg.
        self.nearest_neighbour_transformed_mesh = None  # source mesh transformed to target w/ nearest neighbour
        self.corresponding_target_idx_for_each_source_pt = None  # Final correspondence (target ID for each source pt)
        self.nearest_neighbor_transformed_points = None  # location source points move on target as nearest neighbor.
        self.weighted_avg_transformed_points = None  # location source points move on the target mesh as weighted avg.
        self.average_mesh = None  # average of the two meshes (based on the correspondences).
        # self.nearest_neighbour_transformed_mesh = None
        # self.weighted_avg_transformed_mesh = None
        # NEED ONE MORE OUTSPUTS:
        # MESH REPRESENTING MEAN OF TWO MESHES

    """
    Functions to prepare pointsets to be registered. 
    """

    def append_features_to_spectral_coords(self):
        print('Appending Extra Features to Spectral Coords')
        if self.graph_source.n_extra_features != self.graph_target.n_extra_features:
            raise Exception('Number of extra features between'
                            ' target ({}) and source ({}) dont match!'.format(self.graph_target.n_extra_features,
                                                                              self.graph_source.n_extra_features))

        self.source_extra_features = np.zeros((self.graph_source.n_points, self.graph_source.n_extra_features))
        self.target_extra_features = np.zeros((self.graph_target.n_points, self.graph_target.n_extra_features))

        for feature_idx in range(self.graph_source.n_extra_features):
            self.source_extra_features[:, feature_idx] = self.graph_source.mean_filter_graph(
                self.graph_source.node_features[feature_idx], iterations=self.feature_smoothing_iterations)
            self.source_extra_features[:, feature_idx] = self.source_extra_features[:, feature_idx] \
                                                         - np.min(self.source_extra_features[:, feature_idx])
            self.source_extra_features[:, feature_idx] = self.source_extra_features[:, feature_idx] \
                                                         / np.max(self.source_extra_features[:, feature_idx])
            self.source_extra_features[:, feature_idx] = np.ptp(self.source_spectral_coords) * self.source_extra_features[:, feature_idx]

            self.target_extra_features[:, feature_idx] = self.graph_target.mean_filter_graph(
                self.graph_target.node_features[feature_idx], iterations=self.feature_smoothing_iterations)
            self.target_extra_features[:, feature_idx] = self.target_extra_features[:, feature_idx] \
                                                         - np.min(self.target_extra_features[:, feature_idx])
            self.target_extra_features[:, feature_idx] = self.target_extra_features[:, feature_idx] \
                                                         / np.max(self.target_extra_features[:, feature_idx])
            self.target_extra_features[:, feature_idx] = np.ptp(
                self.target_spectral_coords) * self.target_extra_features[:, feature_idx]

        self.source_spectral_coords = np.concatenate((self.source_spectral_coords,
                                                      self.source_extra_features), axis=1)
        self.target_spectral_coords = np.concatenate((self.target_spectral_coords,
                                                      self.target_extra_features), axis=1)

    def append_pts_to_spectral_coords(self):
        if self.norm_physical_and_spectral is True:
            self.source_spectral_coords = np.concatenate((self.source_spectral_coords,
                                                          self.graph_source.normed_points), axis=1)
            self.target_spectral_coords = np.concatenate((self.target_spectral_coords,
                                                          self.graph_target.normed_points), axis=1)
        elif self.norm_physical_and_spectral is False:
            # If we dont scale everything down to be 0-1, then assume that we scale everything up to be the same
            # dimensions/range as the original image.
            self.source_spectral_coords = np.concatenate((self.source_spectral_coords
                                                          * self.graph_source.mean_pts_scale_range,
                                                          self.graph_source.points), axis=1)
            self.target_spectral_coords = np.concatenate((self.target_spectral_coords
                                                          * self.graph_target.mean_pts_scale_range,
                                                          self.graph_target.points), axis=1)

    def register_target_to_source(self, reg_type='deformable'):
        if reg_type == 'deformable':
            reg = cycpd.deformable_registration(**{
                'X': self.source_spectral_coords[
                     self.graph_source.get_list_rand_idxs(self.n_coords_spectral_registration), :],
                'Y': self.target_spectral_coords[
                     self.graph_target.get_list_rand_idxs(self.n_coords_spectral_registration), :],
                'num_eig': self.non_rigid_n_eigens,
                'max_iterations': self.non_rigid_max_iterations,
                'tolerance': self.non_rigid_tolerance,
                'alpha': self.non_rigid_alpha,
                'beta': self.non_rigid_beta
            }
                                                          )
            _, self.non_rigid_params = reg.register()
        elif reg_type == 'affine':
            # Using affine instead of truly rigid, because rigid doesnt accept >3 dimensions at moment.
            reg = cycpd.affine_registration(**{
                'X': self.source_spectral_coords[
                     self.graph_source.get_list_rand_idxs(self.n_coords_spectral_registration), :],
                'Y': self.target_spectral_coords[
                     self.graph_target.get_list_rand_idxs(self.n_coords_spectral_registration), :],
                'max_iterations': self.rigid_reg_max_iterations,
                'tolerance': self.rigid_tolerance
            }
                                                  )
            _, self.rigid_params = reg.register()

        # Apply transform to all points (ensures all points are transformed even if not all used for registration).
        self.target_spectral_coords = reg.transform_point_cloud(self.target_spectral_coords)

    """
    Functions to find correspondences between arrays of points. 
    """

    def get_hungarian_correspondence(self, target_pts, spectral_pts):
        tic = time.time()
        distances = cdist(spectral_pts, target_pts)
        toc = time.time()
        print('time to get cdist: {}'.format(toc - tic))
        tic = time.time()
        source_idx, target_idx = linear_sum_assignment(distances)
        toc = time.time()
        print('time to linear sum assignment: {}'.format(toc - tic))
        self.corresponding_target_idx_for_each_source_pt = target_idx

    def get_kd_correspondence(self, target_pts, spectral_pts):
        tree = KDTree(target_pts)
        _, self.corresponding_target_idx_for_each_source_pt = tree.query(spectral_pts)

    def get_initial_correspondences(self):
        """
        Find target idx that is closest to each source point.
        The correspondences indicate where (on the target mesh) each source point should move to.
        :return:
        """
        if self.initial_correspondence_type == 'kd':
            self.get_kd_correspondence(self.target_spectral_coords, self.source_spectral_coords)
        elif self.initial_correspondence_type == 'hungarian':
            self.get_hungarian_correspondence(self.target_spectral_coords, self.source_spectral_coords)

    def get_smoothed_correspondences(self):
        # Smooth the XYZ vertices using adjacency matrix for target
        # This will filter the target points using a low-pass filter
        self.smoothed_target_coords = self.graph_target.mean_filter_graph(self.graph_target.points,
                                                                          iterations=self.graph_smoothing_iterations)
        # Next, we take each of these smoothed points (particularly arranged based on which ones best align with
        # the spectral coordinates of the target mesh) and we smooth these vertices/values using the adjacency/degree
        # matrix of the source mesh. I.e. the target mesh coordinates are smoothed on the surface of the source mesh.
        if ((self.smoothed_target_coords.shape[0] != self.graph_source.n_points)
                & (self.initial_correspondence_type == 'hungarian')):
            raise Exception("If number vertices between source & target don't match, initial_correspondence_type must\n"
                            "be 'kd' and not 'hungarian'. Current type is: {}".format(self.initial_correspondence_type))
        self.source_projected_on_target = self.graph_source.mean_filter_graph(self.smoothed_target_coords[self.corresponding_target_idx_for_each_source_pt, :],
                                                                              iterations=self.projection_smooth_iterations)

        if self.final_correspondence_type == 'kd':
            self.get_kd_correspondence(self.smoothed_target_coords, self.source_projected_on_target)
        elif self.final_correspondence_type == 'hungarian':
            self.get_hungarian_correspondence(self.smoothed_target_coords, self.source_projected_on_target)

        # This now matches/makes correspondences. Can use this correspondence.
        # Or can associate with points in between these points...

    def get_weighted_final_node_locations(self, n_closest_pts=3):
        """
        Disperse points (from source) over the target mesh surface - distribute them instead of just finding the
        closest point.
        :return:
        """
        self.weighted_avg_transformed_points = np.zeros_like(self.graph_source.points)

        tree = KDTree(self.smoothed_target_coords)
        for pt_idx in range(self.graph_source.n_points):
            closest_pt_distances, closest_pt_idxs = tree.query(self.source_projected_on_target[pt_idx, :],
                                                               k=n_closest_pts)

            if 0 in closest_pt_distances:
                idx_coincident = np.where(closest_pt_distances == 0)[0][0]
                self.weighted_avg_transformed_points[pt_idx, :] = self.graph_target.points[closest_pt_idxs[idx_coincident]]
            else:
                weighting = 1 / closest_pt_distances[:, None]

                avg_location = np.sum(self.graph_target.points[closest_pt_idxs, :] * weighting, axis=0) / (sum(weighting))
                self.weighted_avg_transformed_points[pt_idx, :] = avg_location

    def get_nearest_neighbour_final_node_locations(self):
        self.nearest_neighbor_transformed_points = self.graph_target.points[self.corresponding_target_idx_for_each_source_pt, :]

    def get_average_shape(self, align_type='weighted'):
        """
        Get new mesh average of the transformed source & target.
        :return:
        """
        self.average_mesh = vtk_deep_copy(self.graph_source.vtk_mesh)

        points = self.average_mesh.GetPoints()
        if align_type == 'nearest':
            for src_pt_idx in range(self.graph_source.n_points):
                trget_pt_idx = self.corresponding_target_idx_for_each_source_pt[src_pt_idx]
                new_xyz = self.graph_target.vtk_mesh.GetPoint(trget_pt_idx)
                orig_xyz = self.graph_source.points[src_pt_idx]
                mean_xyz = (orig_xyz + new_xyz) / 2
                points.SetPoint(src_pt_idx, mean_xyz)
        elif align_type == 'weighted':
            for src_pt_idx in range(self.graph_source.n_points):
                orig_xyz = self.weighted_avg_transformed_points[src_pt_idx]
                new_xyz = self.graph_source.points[src_pt_idx]
                mean_xyz = (orig_xyz + new_xyz) / 2
                points.SetPoint(src_pt_idx, mean_xyz)


    """
    Spectral Weighting
    """

    def calc_c_weighting_spectral(self):
        """
        calculate spectral weighting coefficient. If 10 spectral coorindates (per point), would calculate
        10 weighting coefficients to weight importance of these coordinates.

        c^(u) = exp( -(Q^u * lambda^u)^2 / 2sigma^2)

        lambda^u = uth eigenvalue and represents smoothness of eigenvector (spectral coordinates)
        Q^u = confidence in re-ordered postion of uth set of eigenvactors, eigenvalues - from the dissimilarity
            matrix Q.
        sigma = mean {Q^u * lambda^u} subscript (u=1...m)


        Because this step is finally done with re-ordering etc. We should probably use only the n spectral coords that
        we intend to use for the analysis.



        :return:
        """

        # highest eigenvalue weight from the two meshes (for a given pair).
        self.spectral_weights = self.Q[:self.n_spectral_features] \
                                * np.max((self.graph_source.eig_vals[:self.n_spectral_features],
                                      self.graph_target.eig_vals[:self.n_spectral_features]),
                                      axis=0)
        # Q is a weighting factor to get the mean weight (across the different spectral coordinates).
        sigma = np.mean(self.spectral_weights)
        self.spectral_weights = np.exp(-(self.spectral_weights ** 2) / (2 * sigma ** 2))

    def calc_weighted_spectral_coords(self):
        self.calc_c_weighting_spectral()
        self.source_spectral_coords = self.graph_source.eig_vecs[:, :self.n_spectral_features] \
                                      * self.spectral_weights[None, :]
        self.target_spectral_coords = self.graph_target.eig_vecs[:, :self.n_spectral_features] \
                                      * self.spectral_weights[None, :]

    def calc_spectral_coords(self):
        if self.get_weighted_spectral_coords is True:
            self.calc_weighted_spectral_coords()
        elif self.get_weighted_spectral_coords is False:
            self.source_spectral_coords = self.graph_source.eig_vecs[:, :self.n_spectral_features]
            self.target_spectral_coords = self.graph_target.eig_vecs[:, :self.n_spectral_features]

    """
    Align Maps
    """

    def align_maps(self):
        eig_map_sorter = eigsort(graph_target=self.graph_target,
                                 graph_source=self.graph_source,
                                 n_features=self.n_total_spectral_features,
                                 target_as_reference=self.target_eigenmap_as_reference)
        self.Q = eig_map_sorter.sort_eigenmaps()
        self.calc_spectral_coords()

        if (self.graph_source.n_extra_features > 0) & (self.use_features_as_coords is True):
            self.append_features_to_spectral_coords()

        if self.include_points_as_features is True:
            self.append_pts_to_spectral_coords()

        self.source_spectral_coords_b4_reg = np.copy(self.source_spectral_coords)

        print('Number of features (including spectral) '
              'used for registartion: {}'.format(self.target_spectral_coords.shape[1]))

        if self.rigid_before_non_rigid_reg is True:
            print_header('Rigid Registration Beginning!')
            self.register_target_to_source(reg_type='affine')
            self.source_spectral_coords_after_rigid = np.copy(self.source_spectral_coords)

        print_header('Non-Rigid (Deformable) Registration Beginning')
        self.register_target_to_source('deformable')

        self.get_initial_correspondences()
        print('Number of unique correspondences: {}'.format(len(np.unique(self.corresponding_target_idx_for_each_source_pt))
                                                            ))
        if self.smooth_correspondences is True:
            self.get_smoothed_correspondences()
            print('Number of unique correspondences after smoothing: {}'.format(
                len(np.unique(self.corresponding_target_idx_for_each_source_pt))
                ))

        if self.return_average_final_points is True:
            self.get_weighted_final_node_locations()
        if self.return_nearest_final_points is True:
            self.get_nearest_neighbour_final_node_locations()

        if self.return_transformed_mesh is True:
            if self.return_average_final_points is True:
                self.get_source_mesh_transformed_weighted_avg()
            if self.return_nearest_final_points is True:
                self.get_source_mesh_transformed_nearest_neighbour()

        # return self.corresponding_target_idx_for_each_source_pt

    """
    Change mesh scalar values (for visualizations). 
    """

    def set_transformed_source_scalars_to_corresp_target_idx(self):
        if self.weighted_avg_transformed_mesh is not None:
            self.weighted_avg_transformed_mesh.GetPointData().SetScalars(
            numpy_to_vtk(self.corresponding_target_idx_for_each_source_pt))
        if self.nearest_neighbour_transformed_mesh is not None:
            self.nearest_neighbour_transformed_mesh.GetPointData().SetScalars(
            numpy_to_vtk(self.corresponding_target_idx_for_each_source_pt))

    def set_source_scalars_to_corresp_target_idx(self):
        self.graph_source.vtk_mesh.GetPointData().SetScalars(
            numpy_to_vtk(self.corresponding_target_idx_for_each_source_pt))

    def set_target_scalars_to_corresp_target_idx(self):
        self.graph_target.vtk_mesh.GetPointData().SetScalars(
            numpy_to_vtk(np.arange(self.graph_target.n_points)))

    def set_all_mesh_scalars_to_corresp_target_idx(self):
        self.set_target_scalars_to_corresp_target_idx()
        self.set_source_scalars_to_corresp_target_idx()
        self.set_transformed_source_scalars_to_corresp_target_idx()

    """
    Probing & View Results
    """
    def get_source_mesh_transformed_weighted_avg(self):
        """
        Create new mesh same as source mesh. Get source points. Move source points to transformed location(s) on
        target mesh using weighted average locations.
        :return:
        """
        self.weighted_avg_transformed_mesh = vtk_deep_copy(self.graph_source.vtk_mesh)
        points = self.weighted_avg_transformed_mesh.GetPoints()
        for src_pt_idx in range(self.graph_source.n_points):
            points.SetPoint(src_pt_idx, self.weighted_avg_transformed_points[src_pt_idx])

    def get_source_mesh_transformed_nearest_neighbour(self):
        """
        Create new mesh same as source mesh. Get source points. Move source points to transformed location(s) on
        target mesh using nearest neighbour locations.
        :return:
        """
        self.nearest_neighbour_transformed_mesh = vtk_deep_copy(self.graph_source.vtk_mesh)
        points = self.nearest_neighbour_transformed_mesh.GetPoints()
        for src_pt_idx in range(self.graph_source.n_points):
            points.SetPoint(src_pt_idx, self.nearest_neighbor_transformed_points[src_pt_idx])

    # def get_source_mesh_transformed(self):
    #     """
    #     Create new mesh same as source mesh. Get source points. Move source points to transformed location(s) on
    #     target mesh.
    #     :return:
    #     """
    #     self.source_vtk_mesh_transformed = vtk_deep_copy(self.graph_source.vtk_mesh)
    #     points = self.source_vtk_mesh_transformed.GetPoints()
    #
    #     points = self.source_vtk_mesh_transformed.GetPoints()
    #     if self.diffuse_final_points is False:
    #         for src_pt_idx in range(self.graph_source.n_points):
    #             trget_pt_idx = self.corresponding_target_idx_for_each_source_pt[src_pt_idx]
    #             trget_pt_xyz = self.graph_target.vtk_mesh.GetPoint(trget_pt_idx)
    #             points.SetPoint(src_pt_idx, trget_pt_xyz)
    #     elif self.diffuse_final_points is True:
    #         for src_pt_idx in range(self.graph_source.n_points):
    #             points.SetPoint(src_pt_idx, self.weighted_avg_transformed_points[src_pt_idx])


    def view_aligned_spectral_coords(self, starting_spectral_coord=0,
                                     point_set_representations=['spheres'],
                                     point_set_colors=None,
                                     include_target_coordinates=True,
                                     include_non_rigid_aligned=True,
                                     include_rigid_aligned=False,
                                     include_unaligned=False,
                                     upscale_factor=10.
                                     ):

        point_sets = []

        if include_target_coordinates is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.target_spectral_coords[:, starting_spectral_coord:starting_spectral_coord + 3]))

        if include_unaligned is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.source_spectral_coords_b4_reg[:, starting_spectral_coord:starting_spectral_coord + 3]))

        if include_rigid_aligned is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.source_spectral_coords_after_rigid[:, starting_spectral_coord:starting_spectral_coord + 3]))

        if include_non_rigid_aligned is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.source_spectral_coords[:, starting_spectral_coord:starting_spectral_coord+3]))

        # Make all the same shape, if only one specified and more than one point et included.
        if (len(point_set_representations) == 1) & (len(point_sets) > 1):
            point_set_representations = point_set_representations * len(point_sets)

        # Colours points sets sequentially using matplotlib V2 colours:
        if point_set_colors is None:
            point_set_colors = [colors.to_rgb('C{}'.format(x)) for x in range(len(point_sets))]
        plotter = Viewer(point_sets=point_sets,
                         point_set_representations=point_set_representations,
                         point_set_colors=point_set_colors
                         )

        return plotter

    def view_meshes_colored_by_spectral_correspondences(self,
                                                        x_translation=100,
                                                        y_translation=0,
                                                        z_translation=0,
                                                        shadow=True):
        target_mesh = vtk_deep_copy(self.graph_target.vtk_mesh)
        target_mesh.GetPointData().SetScalars(numpy_to_vtk(np.arange(self.graph_target.n_points)))

        source_mesh = vtk_deep_copy(self.graph_source.vtk_mesh)
        source_mesh.GetPointData().SetScalars(numpy_to_vtk(self.corresponding_target_idx_for_each_source_pt))

        target_transform = vtk.vtkTransform()
        target_transform.Translate(x_translation, y_translation, z_translation)
        target_mesh = apply_transform(target_mesh, target_transform)

        plotter = Viewer(geometries=[source_mesh, target_mesh], shadow=shadow)
        return plotter

    def view_aligned_smoothed_spectral_coords(self):
        plotter = Viewer(point_sets=[self.smoothed_target_coords, self.source_projected_on_target],
                         point_set_colors=[colors.to_rgb('C0'), colors.to_rgb('C1')])
        return plotter

    def view_meshes(self, include_target=True,
                    include_source=True,
                    include_transformed_target=False,
                    include_average=False,
                    shadow=True
                    ):
        geometries = []
        if include_target is True:
            geometries.append(self.graph_target.vtk_mesh)
        if include_source is True:
            geometries.append(self.graph_source.vtk_mesh)
        if include_transformed_target is True:
            if self.weighted_avg_transformed_mesh is not None:
                geometries.append(self.weighted_avg_transformed_mesh)
            elif self.nearest_neighbour_transformed_mesh is not None:
                geometries.append(self.nearest_neighbour_transformed_mesh)
            elif self.weighted_avg_transformed_points is not None:
                self.get_weighted_final_node_locations()
                self.get_source_mesh_transformed_weighted_avg()
                geometries.append(self.weighted_avg_transformed_mesh)
            elif self.nearest_neighbor_transformed_points is not None:
                self.get_nearest_neighbour_final_node_locations()
                self.get_source_mesh_transformed_nearest_neighbour()
                geometries.append(self.nearest_neighbour_transformed_mesh)
            else:
                raise Exception('No corresponding points or meshes calculated. Try running: \n'
                                'reg.get_weighted_final_node_locations()\n'
                                'reg.get_nearest_neighbour_final_node_locations()\n'
                                'or try re-running with the flags: \n'
                                'return_average_final_points=True & return_transformed_mesh=True')
        if include_average is True:
            if self.average_mesh is None:
                if self.weighted_avg_transformed_points is not None:
                    self.get_average_shape()
                elif self.nearest_neighbor_transformed_points is not None:
                    self.get_average_shape(align_type='nearest')
                else:
                    raise Exception("No xyz correspondences calculated can't get average! Try:\n"
                                    "`reg.get_weighted_final_node_locations` or `reg.get_nearest_neighbour_final_node_locations`")
            geometries.append(self.average_mesh)

        plotter = Viewer(geometries=geometries, shadow=shadow)
        return plotter