import numpy as np

from Compute_Similarity_Python import Compute_Similarity_Python
from helper import Helper
from run import Runner


class ContentBasedFiltering(object):

    def __init__(self, knn_alb=100, knn_art=300, shrink_alb=2, shrink_art=2, weight_alb=0.6, use_tail_boost=False):
        self.knn_album = knn_alb
        self.knn_artist = knn_art
        self.shrink_album = shrink_alb
        self.shrink_artist = shrink_art
        self.weight_album = weight_alb
        self.helper = Helper()
        self.use_tail_boost = use_tail_boost

    def compute_similarity(self, ICM, knn, shrink):
        similarity_object = Compute_Similarity_Python(ICM.transpose(), shrink=knn, topK=shrink,
                                                     normalize=True, similarity="cosine")
        return similarity_object.compute_similarity()

    def fit(self, URM):

        self.URM = URM

        if self.use_tail_boost:
            helper = Helper()
            self.URM = helper.tail_boost(self.URM)

        self.ICM_album = self.helper.get_icm_album()
        ### ICM_album PROCESSING
        self.ICM_album = self.helper.get_URM_BM_25(self.ICM_album)
        ########################

        self.ICM_artist = self.helper.get_icm_artist()
        ### ICM_artist PROCESSING
        self.ICM_artist = self.helper.get_URM_tfidf(self.ICM_artist)
        self.ICM_artist = self.helper.get_mat_normalize(self.ICM_artist)

        self.SM_album = self.compute_similarity(self.ICM_album, self.knn_album, self.shrink_album)
        self.SM_artist = self.compute_similarity(self.ICM_artist, self.knn_artist, self.shrink_artist)

    def get_expected_ratings(self, playlist_id):
        playlist_id = int(playlist_id)
        liked_tracks = self.URM[playlist_id]
        expected_ratings_album = liked_tracks.dot(self.SM_album).toarray().ravel()
        expected_ratings_artist = liked_tracks.dot(self.SM_artist).toarray().ravel()

        album_weight = self.weight_album
        artist_weight = 1 - album_weight
        expected_ratings = (expected_ratings_album * album_weight) + (expected_ratings_artist * artist_weight)
        expected_ratings[liked_tracks.indices] = -10
        return expected_ratings

    def recommend(self, playlist_id):
        playlist_id = int(playlist_id)
        expected_ratings = self.get_expected_ratings(playlist_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        return recommended_items[:10]

### CODE TO TEST THIS ALG
# recommender = ContentBasedFiltering(knn_alb = 10,  knn_art = 10, shrink_alb = 0, shrink_art = 0, weight_alb = 0.7)
# Runner.run(is_test = True, recommender = recommender, split_type = None)
#########################