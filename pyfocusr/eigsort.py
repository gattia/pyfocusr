import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

from .main import *


class eigsort(object):
    """
    Class to sort eigen vectors/ spectral coordinates necessary before aligning/registering spectral coordinates.
    """
    def __init__(self,
                 graph_target,
                 graph_source,
                 n_features,
                 target_as_reference=True, # if `True``, then target order used as reference and source permuted
                                           # to match target. If `False`, then source order used as reference and
                                           # target is permuted. 
                 ):

        # Input variables

        self.graph_target = graph_target  # Target mesh (match source eigs to this)
        self.graph_source = graph_source  # Source mesh (eigs to re-order/flip to match target).
        self.n_features = n_features  # number of features (eigenvecs/values) to consider during alignment.
        self.target_as_reference = target_as_reference  # If true, then the target mesh is used as the reference

        # Build/create specified data needed for alignment.

        # Points
        self.rand_target_points = self.graph_target.get_rand_normalized_points()  # Get rand target points for alignment
        self.rand_source_points = self.graph_source.get_rand_normalized_points()  # Get rand source points for alignment
        # Eigenvectors
        self.rand_target_eig_vecs = self.graph_target.get_rand_eig_vecs()  # Get eigvecs of rand target points
        self.rand_source_eig_vecs = self.graph_source.get_rand_eig_vecs()  # Get eigvecs of rand source points

        # Interim values used in class.

        self.c_lambda = np.zeros((self.n_features, self.n_features))  # eigenvalue cost matrix
        self.c_hist = np.zeros_like(self.c_lambda)  # histogram comparison matrix
        self.c_hist_f = np.zeros_like(self.c_lambda)  # flipped histogram cost matrix.
        self.c_spatial = np.zeros_like(self.c_lambda)  # spatial cost matrix
        self.c_spatial_f = np.zeros_like(self.c_lambda)  # flipped spatial cost matrix

        # Returns.
        self.Q = None  # Cost of making each particular "pairing" of eigenvectors/values between two graphs.

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
        if self.target_as_reference is True:
            target_matches, source_matches = linear_sum_assignment(self.Q)
        elif self.target_as_reference is False:
            source_matches, target_matches = linear_sum_assignment(self.Q.T)
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
            if self.target_as_reference is True:
                self.graph_source.eig_vecs[:, mode_1] = self.graph_source.eig_vecs[:, mode_1] * -1
            elif self.target_as_reference is False:
                self.graph_target.eig_vecs[:, mode_0] = self.graph_target.eig_vecs[:, mode_0] * -1
        # The source eig_vecs are re-ordered to match the order from the target eig_vecs - based on the best matches
        # Identified using Hungarian algorithm (linear_sum_assignment) on the dissimilarity matrix Q.
        if self.target_as_reference is True:
            self.graph_source.eig_vecs[:, target_matches] = self.graph_source.eig_vecs[:, source_matches]
        elif self.target_as_reference is False:
            self.graph_target.eig_vecs[:, source_matches] = self.graph_target.eig_vecs[:, target_matches]
        print_header('Eigenvector Sorting Results')
        if self.target_as_reference is True:
            print('Using target eigenmaps as the reference')
        elif self.target_as_reference is False:
            print('Using source eigenmaps as the reference')
        print('The matches for eigenvectors were as follows:')
        print('Target\t|  Source')
        for matched_pair in zip(target_matches, source_matches):
            source_value = str(matched_pair[1])
            target_value = str(matched_pair[0])
            if matched_pair in flipped_pairs:
                if self.target_as_reference is True:
                    source_value = '-' + source_value
                elif self.target_as_reference is False:
                    target_value = '-' + target_value

            print('{:6}\t|  {:6}'.format(target_value, source_value))
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
        for i in range(self.n_features):
            for j in range(self.n_features):
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

        # Initially tried using straight values (not log) but eig vec 1 got accentuated too much (in the weighting)
        # Then tried just log, but becuase there are negative values it creates erro.
        # need to add .5 + a small value to ensure there are no 0 values entered to log.
        # wasserstein_distance is the same as earth movers distance, and is the minimum "work" needed to
        # transform u (first entry) into v (second entry).
        eps = np.finfo(float).eps
        for i in range(self.n_features):
            for j in range(self.n_features):
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

        source_kd_tree = KDTree(self.rand_source_points)
        _, idx_source_for_each_target_pt = source_kd_tree.query(self.rand_target_points)

        # Using the same error as the Matlab code., which is the average euclidean error in n-dimensions.
        # Paper says that it used the sum of squares.
        for i in range(self.n_features):
            for j in range(self.n_features):
                self.c_spatial[i, j] = np.sqrt(np.sum((self.rand_source_eig_vecs[idx_source_for_each_target_pt, j]
                                                      - self.rand_target_eig_vecs[:, i])**2)
                                               )/self.rand_target_eig_vecs.shape[0]
                self.c_spatial_f[i, j] = np.sqrt(np.sum((-self.rand_source_eig_vecs[idx_source_for_each_target_pt, j]
                                                        - self.rand_target_eig_vecs[:, i])**2)
                                                 )/self.rand_target_eig_vecs.shape[0]

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

        return self.Q  # Q was calculated in eigen_sort and is the cost associated with each of the final pairings.
