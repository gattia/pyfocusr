import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .graph import Graph
import cycpd
from itkwidgets import Viewer
from matplotlib import cm
from .vtk_functions import *
import time


class Focusr(object):
    def __init__(self,
                 vtk_mesh_target,
                 vtk_mesh_source,
                 n_spectral_features=3,
                 n_extra_spectral=3,
                 norm_physical_and_spectral=True,
                 n_coords_spectral_ordering=5000,
                 n_coords_spectral_registration=5000,
                 rigid_before_non_rigid_reg=True,
                 rigid_reg_max_iterations=100,
                 rigid_tolerance=1e-8,
                 non_rigid_max_iterations=1000,
                 non_rigid_tolerance=1e-8,
                 non_rigid_alpha=0.5,
                 non_rigid_beta=3.0,
                 non_rigid_n_eigens=100,
                 include_points_as_features=False,
                 get_weighted_spectral_coords=True,
                 list_features_to_calc=['curvature'],
                 graph_smoothing_iterations=300,
                 feature_smoothing_iterations=40,
                 smooth_correspondences=True,
                 projection_smooth_iterations=40,
                 feature_weights=None,
                 initial_correspondence_type='kd',
                 final_correspondence_type='hungarian'):

        icp = icp_transform(target=vtk_mesh_target, source=vtk_mesh_source)
        vtk_mesh_source = apply_transform(source=vtk_mesh_source, transform=icp)

        self.corresponding_target_idx_for_each_source_pt = None
        self.initial_correspondence_type = initial_correspondence_type
        self.final_correspondence_type = final_correspondence_type

        # either normalize all (spectral & points) to be in same range.
        # Or, normalize the spectral and other values to be in same range as the physical coordinates.
        self.norm_physical_and_spectral = norm_physical_and_spectral

        self.n_coords_spectral_registration = n_coords_spectral_registration
        self.n_spectral_features = n_spectral_features
        self.n_extra_spectral = n_extra_spectral
        self.n_total_spectral_features = self.n_spectral_features + self.n_extra_spectral

        self.graph_target = Graph(vtk_mesh_target,
                                  n_spectral_features=self.n_total_spectral_features,
                                  n_rand_samples=n_coords_spectral_ordering,
                                  list_features_to_calc=list_features_to_calc,
                                  feature_weights=feature_weights
                                  )
        print('Loaded Mesh 1')
        self.graph_target.get_graph_spectrum()
        print('Computed spectrum 1')
        self.graph_source = Graph(vtk_mesh_source,
                                  n_spectral_features=self.n_total_spectral_features,
                                  n_rand_samples=n_coords_spectral_ordering,
                                  list_features_to_calc=list_features_to_calc,
                                  feature_weights=feature_weights
                                  )
        print('Loaded Mesh 2')
        self.graph_source.get_graph_spectrum()
        print('Computed spectrum 2')

        self.rand_target_eig_vecs = None
        self.rand_source_eig_vecs = None

        self.rand_target_points = None
        self.rand_source_points = None

        self.c_lambda = np.zeros((self.n_total_spectral_features, self.n_total_spectral_features))
        self.c_hist = np.zeros_like(self.c_lambda)
        self.c_hist_f = np.zeros_like(self.c_lambda)
        self.c_spatial = np.zeros_like(self.c_lambda)
        self.c_spatial_f = np.zeros_like(self.c_lambda)
        self.Q = None
        self.spec_weights = None
        self.source_spectral_coords = None  # Could pre-allocate using np.zeros()
        self.target_spectral_coords = None  # Could pre-allocate using np.zeros()

        self.get_weighted_spectral_coords = get_weighted_spectral_coords

        self.include_points_as_features = include_points_as_features

        self.source_extra_features = None # Extra features used for mapping
        self.target_extra_features = None

        self.rigid_before_non_rigid_reg = rigid_before_non_rigid_reg
        self.rigid_reg_max_iterations = rigid_reg_max_iterations
        self.rigid_tolerance = rigid_tolerance

        self.non_rigid_max_iterations = non_rigid_max_iterations
        self.non_rigid_tolerance = non_rigid_tolerance
        self.non_rigid_alpha = non_rigid_alpha
        self.non_rigid_beta = non_rigid_beta
        self.non_rigid_n_eigens = non_rigid_n_eigens

        self.source_spectral_coords_after_rigid = None
        self.source_spectral_coords_b4_reg = None
        self.rigid_params = None
        self.non_rigid_params = None

        self.graph_smoothing_iterations = graph_smoothing_iterations
        self.feature_smoothing_iterations = feature_smoothing_iterations
        self.smooth_correspondences = smooth_correspondences
        self.projection_smooth_iterations = projection_smooth_iterations

        self.smoothed_target_coords = None
        self.source_projected_on_target = None

        self.source_vtk_mesh_transformed = None

    """
    Functions used to sort eigenvectors. 
    """

    def eigen_sort(self):
        """
        - Calculate a metric of similarity c for straight matches, and matches from flipped eigenvectors c_f for source
            mesh.
        - Get the dissimilarity matrix Q by taking the min of c and c_f
        - Get matrix S of positions which were flipped to get minimum
        - Get list of `row_flipped` and `cols_flipped`
        - Get list of matching columns for each row by using hungarian algorithm (linear_sum_assignment) on Q
        - From the matches, and the identified flipped rows/cols, identify which of the pairs are flipped
        - Flip the eigen vector values for the necessary eigen vectors on the source mesh
        """

        # find best matching (negative/positive)
        c = self.c_spatial * self.c_lambda * self.c_hist
        c_f = self.c_spatial_f * self.c_lambda * self.c_hist_f
        self.Q = np.min((c, c_f), axis=0)
        # telling us if sign flipped or not. True = same sign, False = flipped
        S = c > c_f
        (target_flipped, source_flipped) = np.where(S == True)

        # Get matches
        target_matches, source_matches = linear_sum_assignment(self.Q)
        # The original Matlab code calculates Q as:
        # Q = sum(M'*Q')'; which is equivalent to np.sum(np.matmul(M.T, Q.T), axis=0)
        # Where M = nXn matrix where n is number of spectral features (or the length of target_matches). and
        # has 1s where there is a "match", i.e.
        # M = np.zeros((len(target_matches), len(target_matches)))
        # M[target_matches, source_matches] = 1
        # This results in a permutation cost equal to all of the "costs" for the entire row. This seems wrong,
        # it seems like the cost of using a particular "pair" should come down to the cost of just that particular
        # match and not all of the matches that it "didnt" use. In fact, bad matches would have high numbers and screw
        # things up. We are therefore doing the below.
        self.Q = self.Q[target_matches, source_matches]

        # This will identify the source & target eigen #s (pairs after matching) that need to be flipped.
        # So, if we're on eigen 3 for target, but its pair is eigen 5, then this will identify the pair (3, 5).
        # The for loop then iterates through and turns the 5 (from the source) to be negative and leave it where
        # it is. Finally, we will then select the final eigenvecs in the appropriate order (after flipping)
        # using their indices from target_matches, source_matches.

        # If a "flipped pair" is also a "best match", then store the pairs in flipped_pairs so that the corresponding
        # source eig_vecs can be flipped to match the target eig_vec orientation.

        flipped_pairs = [p2 for p1 in zip(target_flipped, source_flipped) for p2 in zip(target_matches, source_matches) if
                         p2 == p1]
        # Flipped pairs are used to flip the sign of the source eig_vecs.

        for mode_0, mode_1 in flipped_pairs:
            self.graph_source.eig_vecs[:, mode_1] = self.graph_source.eig_vecs[:, mode_1] * -1
        # The source eig_vecs are re-ordered to match the order from the target eig_vecs - based on the best matches
        # Identified using Hungarian algorithm (linear_sum_assignment) on the dissimilarity matrix Q.
        self.graph_source.eig_vecs[:, target_matches] = self.graph_source.eig_vecs[:, source_matches]

        print_header('Eigenvector Sorting Results')
        print('The matches for eigenvectors were as follows:')
        print('Target\t|  Source')
        for matched_pair in zip(target_matches, source_matches):
            if matched_pair in flipped_pairs:
                source_value = '-' + str(matched_pair[1])
            else:
                source_value = str(matched_pair[1])
            print('{:6}\t|  {:6}'.format(matched_pair[0], source_value))
        print('*Negative source values means those eigenvectors were flipped*\n ')

    def calc_c_lambda(self):
        """
        Get average gap between successive eigenvalues (used to normalize any errors/differences between
        the eigenvalues of the two meshes.
        :return:
        """

        for graph in [self.graph_source, self.graph_target]:
            if graph.eig_val_gap is None:
                graph.get_eig_val_gap()
        # average the gap of mesh 1 and mesh 2.
        eigen_gap = (self.graph_target.eig_val_gap + self.graph_source.eig_val_gap)/2
        # Calculate difference in eigenvalues between meshes normalizing to eigen_gap.
        for i in range(self.n_total_spectral_features):
            for j in range(self.n_total_spectral_features):
                self.c_lambda[i, j] = np.exp((self.graph_target.eig_vals[i] - self.graph_source.eig_vals[j]) ** 2 /
                                             (2 * eigen_gap ** 2)
                                             )

    def calc_c_hist(self):
        """
        Compare the histograms of all of the eigenvectors (eigenfunctions) for each of the
        meshes - only upto the number of meshes we are interested in. The paper says to do
        this for the number of features we are interested (i.e. 5 or 6), however, this implementation
        does that for more than the requested number of features to ensure that we dont get the wrong ones.
        So, we'll get more spectral coordinates, order them, and then select the appropriate number from
        these re-ordered coordinates.

        :return:
        """

        if self.rand_target_eig_vecs is None:
            self.rand_target_eig_vecs = self.graph_target.get_rand_eig_vecs()
        if self.rand_source_eig_vecs is None:
            self.rand_source_eig_vecs = self.graph_source.get_rand_eig_vecs()

        # Initially tried using straight values (not log) but eig vec 1 got accentuated too much (in the weighting)
        # Then tried just log, but becuase there are negative values it creates erro.
        # need to add .5 + a small value to ensure there are no 0 values entered to log.
        # wasserstein_distance is the same as earth movers distance, and is the minimum "work" needed to
        # transform u (first entry) into v (second entry).
        eps = np.finfo(float).eps
        for i in range(self.n_total_spectral_features):
            for j in range(self.n_total_spectral_features):
                self.c_hist[i, j] = wasserstein_distance(np.log(self.rand_target_eig_vecs[:, i]
                                                                + 0.5
                                                                + eps),
                                                         np.log(self.rand_source_eig_vecs[:, j]
                                                                + 0.5
                                                                + eps))
                self.c_hist_f[i, j] = wasserstein_distance(np.log(self.rand_target_eig_vecs[:, i]
                                                                  + 0.5
                                                                  + eps),
                                                           np.log(-self.rand_source_eig_vecs[:, j]
                                                                  + 0.5
                                                                  + eps))

    def calc_c_spatial(self):
        """
        Next, we calculate the spectral distance of points that are nearest to one another on the mesh.
        This must assume that the meshes are crudely aligned. This will mean that for knees, we will
        always have to flip any left knees to match right knees, we should then use an ICP algoritm to
        get a crude starting alignment. However, even without the ICP (as long as left are flipped)
        this algorithm scales the sizes of the bounding boxes for the points, so they should be very
        crudely aligned.

        :return:
        """

        # Test to see if random eig_vecs/points have been created yet. If not, create them.
        if self.rand_target_points is None:
            self.rand_target_points = self.graph_target.get_rand_normalized_points()
        if self.rand_source_points is None:
            self.rand_source_points = self.graph_source.get_rand_normalized_points()
        if self.rand_target_eig_vecs is None:
            self.rand_target_eig_vecs = self.graph_target.get_rand_eig_vecs()
        if self.rand_source_eig_vecs is None:
            self.rand_source_eig_vecs = self.graph_source.get_rand_eig_vecs()

        source_kd_tree = KDTree(self.rand_source_points)
        _, idx_source_for_each_target_pt = source_kd_tree.query(self.rand_target_points)

        # Using the same error as the Matlab code., which is the average euclidean error in n-dimensions.
        # Paper says that it used the sum of squares.
        for i in range(self.n_total_spectral_features):
            for j in range(self.n_total_spectral_features):
                self.c_spatial[i, j] = np.sqrt(np.sum((self.rand_source_eig_vecs[idx_source_for_each_target_pt, j]
                                                      - self.rand_target_eig_vecs[:, i])**2)
                                               )/self.rand_target_eig_vecs.shape[0]
                self.c_spatial_f[i, j] = np.sqrt(np.sum((-self.rand_source_eig_vecs[idx_source_for_each_target_pt, j]
                                                        - self.rand_target_eig_vecs[:, i])**2)
                                                 )/self.rand_target_eig_vecs.shape[0]

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
        self.spec_weights = self.Q[:self.n_spectral_features] \
                            * np.max((self.graph_source.eig_vals[:self.n_spectral_features],
                                      self.graph_target.eig_vals[:self.n_spectral_features]),
                                     axis=0)
        # Q is a weighting factor to get the mean weight (across the different spectral coordinates).
        sigma = np.mean(self.spec_weights)
        self.spec_weights = np.exp(-(self.spec_weights**2)/(2*sigma**2))

    def calc_weighted_spectral_coords(self):
        self.calc_c_weighting_spectral()
        self.source_spectral_coords = self.graph_source.eig_vecs[:, :self.n_spectral_features] \
                                      * self.spec_weights[None, :]
        self.target_spectral_coords = self.graph_target.eig_vecs[:, :self.n_spectral_features] \
                                      * self.spec_weights[None, :]

    def calc_spectral_coords(self):
        if self.get_weighted_spectral_coords is True:
            self.calc_weighted_spectral_coords()
        elif self.get_weighted_spectral_coords is False:
            self.source_spectral_coords = self.graph_source.eig_vecs[:, :self.n_spectral_features]
            self.target_spectral_coords = self.graph_target.eig_vecs[:, :self.n_spectral_features]

    def sort_eigenmaps(self):
        """
        Run functions necessary to sort eigenvalues.
        Seems to do a good job identifying flips.
        However, if there are abnormal eigenmaps it doesnt do very well.
        :return:
        """
        self.calc_c_lambda()  # lambda = eigenvalues.
        self.calc_c_hist()
        self.calc_c_spatial()
        self.eigen_sort()

    """
    Functions to prepare pointsets to be registered. 
    """

    def append_features_to_spectral_coords(self):
        print('Appending Extra Features to Spectral Coords')
        if self.graph_source.n_features != self.graph_target.n_features:
            raise ('Number of extra features between'
                   ' target ({}) and source ({}) dont match!'.format(self.graph_target.n_features,
                                                                     self.graph_source.n_features))

        self.source_extra_features = np.zeros((self.graph_source.n_points, self.graph_source.n_features))
        self.target_extra_features = np.zeros((self.graph_target.n_points, self.graph_target.n_features))

        for feature_idx in range(self.graph_source.n_features):
            self.source_extra_features[:, feature_idx] = self.graph_source.mean_filter_graph(
                self.graph_source.node_features[feature_idx], iterations=self.feature_smoothing_iterations)
            self.target_extra_features[:, feature_idx] = self.graph_target.mean_filter_graph(
                self.graph_target.node_features[feature_idx], iterations=self.feature_smoothing_iterations)

        self.source_spectral_coords = np.concatenate((self.source_spectral_coords,
                                                      self.source_extra_features), axis=1)
        self.target_spectral_coords = np.concatenate((self.target_spectral_coords,
                                                      self.target_extra_features), axis=1)

    def append_pts_to_spectral_coords(self):
        if self.norm_physical_and_spectral is True:
            self.source_spectral_coords = np.concatenate((self.source_spectral_coords,
                                                          self.graph_source.norm_points), axis=1)
            self.target_spectral_coords = np.concatenate((self.target_spectral_coords,
                                                          self.graph_target.norm_points), axis=1)
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

        _, self.reg_params = reg.register()
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
        if self.initial_correspondence_type is 'kd':
            self.get_kd_correspondence(self.target_spectral_coords, self.source_spectral_coords)
        elif self.initial_correspondence_type is 'hungarian':
            self.get_hungarian_correspondence(self.target_spectral_coords, self.source_spectral_coords)

    def get_smoothed_correspondences(self):
        # Smooth the XYZ vertices using adjacency matrix for target
        # This will filter the target points using a low-pass filter
        self.smoothed_target_coords = self.graph_target.mean_filter_graph(self.graph_target.points, iterations=self.graph_smoothing_iterations)
        # Next, we take each of these smoothed points (particularly arranged based on which ones best align with
        # the spectral coordinates of the target mesh) and we smooth these vertices/values using the adjacency/degree
        # matrix of the source mesh. I.e. the target mesh coordinates are smoothed on the surface of the source mesh.
        self.source_projected_on_target = self.graph_source.mean_filter_graph(self.smoothed_target_coords[self.corresponding_target_idx_for_each_source_pt, :],
                                                                              iterations=self.projection_smooth_iterations)

        if self.final_correspondence_type is 'kd':
            self.get_kd_correspondence(self.smoothed_target_coords, self.source_projected_on_target)
        elif self.final_correspondence_type is 'hungarian':
            self.get_hungarian_correspondence(self.smoothed_target_coords, self.source_projected_on_target)

        # This now matches/makes correspondences. Can use this correspondence.
        # Or can associate with points in between these points...

    def set_position_centroid_closest_n_points(self):
        """
        Interpolate position(s) of final coordinates of source mesh on the surface (between points) on the target mesh.

        Project onto plane created by three closest points?
        :return:
        """
        tree = KDTree(self.smoothed_target_coords)

    """
    Align Maps
    """

    def align_maps(self):
        self.sort_eigenmaps()
        self.calc_spectral_coords()

        if self.graph_source.n_features > 0:
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

        return self.corresponding_target_idx_for_each_source_pt

    """
    Probing & View Results
    """

    def get_source_mesh_transformed(self):
        """
        Create new mesh same as source mesh. Get source points. Move source points to transformed location(s) on
        target mesh.
        :return:
        """
        self.source_vtk_mesh_transformed = vtk_deep_copy(self.graph_source.vtk_mesh)

        points = self.source_vtk_mesh_transformed.GetPoints()
        for src_pt_idx in range(self.graph_source.n_points):
            trget_pt_idx = self.corresponding_target_idx_for_each_source_pt[src_pt_idx]
            trget_pt_xyz = self.graph_target.vtk_mesh.GetPoint(trget_pt_idx)
            points.SetPoint(src_pt_idx, trget_pt_xyz)

    def view_aligned_spectral_coords(self, starting_spectral_coord=0,
                                     colour_idx=False,
                                     point_set_representations=['spheres'],
                                     point_set_colors=None,
                                     include_target_coordinates=True,
                                     include_non_rigid_aligned=True,
                                     include_rigid_aligned=False,
                                     include_unaligned=False,
                                     upscale_factor=10.
                                     ):
        if colour_idx is True:
            cmap = cm.get_cmap('viridis')
            # Use logic to colour each node based on the idx that it is using from target mesh.
            # So, would just be the idx number for the target mesh, but would be the values in
            # self.corresponding_target_idx_for_each_source_pt for the source points.

        else:
            pass
            # point_set_colors = []

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
            point_set_representations = point_set_representations  * len(point_sets)

        # Colours points sets sequentially using matplotlib V2 colours:
        if point_set_colors is None:
            # point_set_colors = ['C{}'.format(x) for x in range(len(point_sets))]
            point_set_colors = tableau_colours[:len(point_sets)]
        plotter = Viewer(point_sets=point_sets,
                         point_set_representations=point_set_representations,
                         point_set_colors=point_set_colors
                         )

        return plotter

    def view_meshes_colored_by_spectral_correspondences(self, x_translation=100, y_translation=0, z_translation=0):
        target_mesh = vtk_deep_copy(self.graph_target.vtk_mesh)
        target_mesh.GetPointData().SetScalars(numpy_to_vtk(np.arange(self.graph_target.n_points)))

        source_mesh = vtk_deep_copy(self.graph_source.vtk_mesh)
        source_mesh.GetPointData().SetScalars(numpy_to_vtk(self.corresponding_target_idx_for_each_source_pt))

        target_transform = vtk.vtkTransform()
        target_transform.Translate(x_translation, y_translation, z_translation)
        target_mesh = apply_transform(target_mesh, target_transform)

        plotter = Viewer(geometries=[source_mesh, target_mesh])
        return plotter

    def view_aligned_smoothed_spectral_coords(self):
        plotter = Viewer(point_sets=[self.smoothed_target_coords, self.source_projected_on_target])
        return plotter

    def set_transformed_source_scalars_to_corresp_target_idx(self):
        self.source_vtk_mesh_transformed.GetPointData().SetScalars(
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

    def view_meshes(self, include_target=True,
                    include_source=True,
                    include_transformed_target=False,
                    ):
        geometries = []
        if include_target is True:
            geometries.append(self.graph_target.vtk_mesh)
        if include_source is True:
            geometries.append(self.graph_source.vtk_mesh)
        if include_transformed_target is True:
            geometries.append(self.source_vtk_mesh_transformed)

        plotter = Viewer(geometries=geometries)
        return plotter

def print_header(message, banner_length=72):
    print('=' * banner_length)
    print('')
    print(message)
    print('')
    print('=' * banner_length)


tableau_colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                   'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                   'tab:olive', 'tab:cyan']