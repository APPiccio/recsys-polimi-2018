from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from helper import Helper
from run import Runner

"""
Recommender with SVD: Singular Value Decomposition technique applied to 
the item content matrix. 

    * k: number of latent factors
    * knn: k-nearest-neighbours to evaluate similarity

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class SVDRec(object):

    def __init__(self, n_factors = 2000, knn = 100, use_tail_boost=False):
        self.n_factors = n_factors
        self.knn = knn
        self.use_tail_boost = use_tail_boost

    def fit(self, URM):
        self.URM = URM

        if self.use_tail_boost:
            helper = Helper()
            self.URM = helper.tail_boost(self.URM)

        print("Getting deafult path")
        path_to_icm = "../data/icm_svd.npy"
        icm_file = Path(path_to_icm)
        if icm_file.is_file():
            print("Loading icm from file " + path_to_icm)
            self.S_ICM_SVD = np.load(path_to_icm)
        else:
            print("No icm found in " + path_to_icm + ". Generating a new one, this will take a while..")
            helper = Helper()
            self.ICM = helper.get_icm()
            self.S_ICM_SVD = self.get_S_ICM_SVD(self.ICM, k=self.n_factors, knn=self.knn)
            np.save('../data/icm_svd.npz.npy', self.S_ICM_SVD)


    def get_expected_ratings(self, playlist_id):
        playlist_id = int(playlist_id)
        liked_tracks = self.URM[playlist_id]
        expected_ratings = liked_tracks.dot(self.S_ICM_SVD)
        return expected_ratings[0]

    def recommend(self, playlist_id, at=10):
        playlist_id = int(playlist_id)
        expected_ratings = self.get_expected_ratings(playlist_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[playlist_id].indices,
                                        assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

    def get_S_ICM_SVD(self, ICM, k, knn):
        print('Computing S_ICM_SVD...')

        S_matrix_list = []
        ICM = ICM.astype(np.float64)
        ICM = sparse.csr_matrix(ICM)
        u, s, vt = svds(ICM, k=k, which='LM')

        ut = u.T

        s_2_flatten = np.power(s, 2)
        s_2 = np.diagflat(s_2_flatten)
        s_2_csr = sparse.csr_matrix(s_2)

        S = u.dot(s_2_csr.dot(ut))

        # for i in tqdm(range(0, u.shape[0])):
        #     S_row = u[i].dot(s_2_csr.dot(ut))
        #     r = S_row.argsort()[:-knn]
        #     S_row[r] = 0
        #     S_matrix_list.append(S_row)
        #
        # S = sc.sparse.vstack(S_matrix_list)

        return S

# recommender = SVDRec()
# Runner.run(True, recommender, split_type=None)
