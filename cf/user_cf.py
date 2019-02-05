

import numpy as np
from Compute_Similarity_Python import Compute_Similarity_Python
from helper import Helper
from run import Runner


class UserBasedCollaborativeFiltering(object):
    def __init__(self, knn=300, shrink=4, similarity="cosine", use_tail_boost=False):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.URM = None
        self.SM_user = None
        self.use_tail_boost = use_tail_boost

    def generate_similarity_matrix(self):
        similarity_object = Compute_Similarity_Python(self.URM.transpose(), self.knn, self.shrink, True,
                                                      self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM):
        helper = Helper()
        self.URM = URM
        if self.use_tail_boost:
            helper = Helper()
            self.URM = helper.tail_boost(self.URM)
        self.SM_user = self.generate_similarity_matrix()
        self.RECS = self.SM_user.dot(self.URM)

    def get_expected_ratings(self, playlist_id):
        playlist_id = int(playlist_id)
        expected_ratings = self.RECS[playlist_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, playlist_id, at=10):
        playlist_id = int(playlist_id)
        expected_ratings = self.get_expected_ratings(playlist_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[playlist_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

##### CODE TO TEST THIS ALG #####
# recommender = UserBasedCollaborativeFiltering(knn = 100, shrink = 0)
# Runner.run(is_test = True, recommender = recommender, split_type = None)
#################################