import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from .graph import Graph
import cycpd
from itkwidgets import Viewer
from matplotlib import cm
from .vtk_functions import *


class Focusr(object):
    def __init__(self,
                 vtk_mesh_target,
                 vtk_mesh_source,
                 n_spectral_features=3,
                 n_extra_spectral=3,
                 norm_physical_and_spectral=True,
                 n_coords_spectral_matching=10000,
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
                 smooth_correspondences=True,
                 projection_smooth_iterations=40,
                 feature_weights=None):

        icp = icp_transform(target=vtk_mesh_target, source=vtk_mesh_source)
        vtk_mesh_source = apply_transform(source=vtk_mesh_source, transform=icp)

        self.corresponding_target_idx_for_each_source_pt = None

        # either normalize all (spectral & points) to be in same range. Or, normalize the spectral and other values
        # to be in same range as the physical coordinates.
        self.norm_physical_and_spectral = norm_physical_and_spectral

        self.n_spectral_features = n_spectral_features
        self.n_extra_spectral = n_extra_spectral
        self.n_total_spectral_features = self.n_spectral_features + self.n_extra_spectral

        self.graph_target = Graph(vtk_mesh_target,
                                  n_spectral_features=self.n_total_spectral_features,
                                  n_rand_samples=n_coords_spectral_matching,
                                  list_features_to_calc=list_features_to_calc,
                                  feature_weights=feature_weights
                                  )
        print('Loaded Mesh 1')
        self.graph_target.get_graph_spectrum()
        print('Loaded spectrum 1')
        self.graph_source = Graph(vtk_mesh_source,
                                  n_spectral_features=self.n_total_spectral_features,
                                  n_rand_samples=n_coords_spectral_matching,
                                  list_features_to_calc=list_features_to_calc,
                                  feature_weights=feature_weights
                                  )
        print('Loaded Mesh 2')
        self.graph_source.get_graph_spectrum()
        print('Loaded spectrum 2')

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

        self.source_spectral_coords_before_non_rigid = None
        self.source_spectral_coords_before_any_reg = None
        self.rigid_params = None
        self.non_rigid_params = None

        self.graph_smoothing_iterations = graph_smoothing_iterations
        self.smooth_correspondences = smooth_correspondences
        self.projection_smooth_iterations = projection_smooth_iterations

        self.smoothed_target_coords = None
        self.source_projected_on_target = None

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

        print('=' * 72)
        print('Eigenvector Sorting Results')
        print('')
        print('The matches for eigenvectors were as follows:')
        print('Target\t|  Source')
        for matched_pair in zip(target_matches, source_matches):
            if matched_pair in flipped_pairs:
                source_value = '-' + str(matched_pair[1])
            else:
                source_value = str(matched_pair[1])
            print('{:6}\t|  {:6}'.format(matched_pair[0], source_value))
        print('*Negative source values means those eigenvectors were flipped*')
        print('')


    def calc_c_lambda(self):
        # Get average gap between successive eigenvalues (used to normalize any errors/differences between
        # the eigenvalues of the two meshes.
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

    def get_extra_features(self):
        """
        Make function to extract/calculate specific features from vtk mesh?
        - Assign them to a features part of the focusr?
        - Have option to just assign them when building focusr instead of calculating after the fact?

        - Normalize to be the same range that spectral coords are
            - so, to start just subtract min and divide by range of data.
            - then, if the spectral get multiplied up to the range of the XYZ position (points) data, then also apply
            the same transformation to these extra features.
        :return:
        """
        return

    def align_maps(self):
        self.calc_c_lambda()  # lambda = eigenvalues.
        self.calc_c_hist()
        self.calc_c_spatial()
        self.eigen_sort()
        self.calc_spectral_coords()
        self.get_extra_features()  # Doesnt do anything right now.

        if self.graph_source.n_features > 0:
            print('Appending Extra Features to Spectral Coords')
            if self.graph_source.n_features != self.graph_target.n_features:
                raise('Number of extra features between'
                      ' target ({}) and source ({}) dont match!'.format(self.graph_target.n_features,
                                                                        self.graph_source.n_features))
            self.source_extra_features = np.zeros((self.graph_source.n_points, self.graph_source.n_features))
            self.target_extra_features = np.zeros((self.graph_target.n_points, self.graph_target.n_features))
            for feature_idx in range(self.graph_source.n_features):
                self.source_extra_features[:, feature_idx] = self.graph_source.mean_filter_graph(self.graph_source.node_features[feature_idx], iterations=self.graph_smoothing_iterations)
                self.target_extra_features[:, feature_idx] = self.graph_target.mean_filter_graph(self.graph_target.node_features[feature_idx], iterations=self.graph_smoothing_iterations)

            self.source_spectral_coords = np.concatenate((self.source_spectral_coords,
                                                          self.source_extra_features), axis=1)
            self.target_spectral_coords = np.concatenate((self.target_spectral_coords,
                                                          self.target_extra_features), axis=1)
        if self.include_points_as_features is True:
            if self.norm_physical_and_spectral is True:
                self.source_spectral_coords = np.concatenate((self.source_spectral_coords,
                                                              self.graph_source.norm_points), axis=1)
                self.target_spectral_coords = np.concatenate((self.target_spectral_coords,
                                                              self.graph_target.norm_points), axis=1)
            elif self.norm_physical_and_spectral is False:
                # If we dont scale everything down to be 0-1, then assume that we scale everything up to be the same
                # dimensions/range as the original image.
                self.source_spectral_coords = np.concatenate((self.source_spectral_coords \
                                                              * self.graph_source.max_points_range,
                                                              self.graph_source.points), axis=1)
                self.target_spectral_coords = np.concatenate((self.target_spectral_coords \
                                                              * self.graph_target.max_points_range,
                                                              self.graph_target.points), axis=1)

        self.source_spectral_coords_before_any_reg = np.copy(self.source_spectral_coords)

        print('Number of features (including spectral) used for registartion: {}'.format(self.target_spectral_coords.shape[1]))

        if self.rigid_before_non_rigid_reg is True:
            print('='*72)
            print('')
            print('Rigid Registration Beginning')
            print('')
            print('=' * 72)
            # Using affine instead of truly rigid, because rigid doesnt accept >3 dimensions at moment.
            rigid_reg = cycpd.affine_registration(**{'X': self.target_spectral_coords,
                                                     'Y': self.source_spectral_coords,
                                                     'max_iterations': self.rigid_reg_max_iterations,
                                                     'tolerance': self.rigid_tolerance
                                                     }
                                                  )
            self.source_spectral_coords, self.rigid_params = rigid_reg.register()

        self.source_spectral_coords_before_non_rigid = np.copy(self.source_spectral_coords)

        print('=' * 72)
        print('')
        print('Non-Rigid (Deformable) Registration Beginning')
        print('')
        print('=' * 72)
        non_rigid_reg = cycpd.deformable_registration(**{'X': self.target_spectral_coords,
                                                         'Y': self.source_spectral_coords,
                                                         'num_eig': self.non_rigid_n_eigens,
                                                         'max_iterations': self.non_rigid_max_iterations,
                                                         'tolerance': self.non_rigid_tolerance,
                                                         'alpha': self.non_rigid_alpha,
                                                         'beta': self.non_rigid_beta
                                                         }
                                                      )
        self.source_spectral_coords, self.non_rigid_params = non_rigid_reg.register()

        self.get_initial_correspondences()
        if self.smooth_correspondences is True:
            self.get_smoothed_correspondences()

        return self.corresponding_target_idx_for_each_source_pt

    def get_initial_correspondences(self):
        """
        Find target idx that is closest to each source point.
        The correspondences indicate where (on the target mesh) each source point should move to.
        :return:
        """
        tree = KDTree(self.target_spectral_coords)
        _, self.corresponding_target_idx_for_each_source_pt = tree.query(self.source_spectral_coords)

    def get_smoothed_correspondences(self):
        # Smooth the XYZ vertices using adjacency matrix for target
        # This will filter the target points using a low-pass filter
        self.smoothed_target_coords = self.graph_target.mean_filter_graph(self.graph_target.points, iterations=self.graph_smoothing_iterations)
        # Next, we take each of these smoothed points (particularly arranged based on which ones best align with
        # the spectral coordinates of the target mesh) and we smooth these vertices/values using the adjacency/degree
        # matrix of the source mesh. I.e. the target mesh coordinates are smoothed on the surface of the source mesh.
        self.source_projected_on_target = self.graph_source.mean_filter_graph(self.smoothed_target_coords[self.corresponding_target_idx_for_each_source_pt, :],
                                                                              iterations=self.projection_smooth_iterations)
        tree = KDTree(self.smoothed_target_coords)
        _, self.corresponding_target_idx_for_each_source_pt = tree.query(self.source_projected_on_target)

        # This now matches/makes correspondences. Can use this correspondnece. Or can associate with points inbetween
        # these points...

    def view_aligned_spectral_coords(self, starting_spectral_coord=0,
                                     colour_idx=False,
                                     point_set_representations=['spheres', 'spheres'],
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
            point_set_colors = []

        point_sets = []

        if include_target_coordinates is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.target_spectral_coords[:, starting_spectral_coord:starting_spectral_coord + 3]))

        if include_non_rigid_aligned is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.source_spectral_coords[:, starting_spectral_coord:starting_spectral_coord+3]))

        if include_rigid_aligned is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.source_spectral_coords_before_non_rigid[:, starting_spectral_coord:starting_spectral_coord + 3]))

        if include_unaligned is True:
            point_sets.append(upscale_factor
                              * np.ascontiguousarray(self.source_spectral_coords_before_any_reg[:, starting_spectral_coord:starting_spectral_coord + 3]))

        plotter = Viewer(point_sets=point_sets,
                         point_set_representations=point_set_representations,
                         point_set_colors=point_set_colors
                         )

        return plotter

    def view_meshes_colored_by_spectral_correspondences(self, x_translation=100, y_translation=0, z_translation=0):
        target_mesh = self.graph_target.vtk_mesh
        target_mesh.GetPointData().SetScalars(numpy_to_vtk(self.corresponding_target_idx_for_each_source_pt))

        source_mesh = self.graph_source.vtk_mesh
        source_mesh.GetPointData().SetScalars(numpy_to_vtk(np.arange(self.graph_source.n_points)))

        target_transform = vtk.vtkTransform()
        target_transform.Translate(x_translation, y_translation, z_translation)
        target_mesh = apply_transform(target_mesh, target_transform)

        plotter = Viewer(geometries=[source_mesh, target_mesh])
        return plotter

