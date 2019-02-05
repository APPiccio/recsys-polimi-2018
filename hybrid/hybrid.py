import pandas as pd
import time

from tqdm import tqdm

from SLIM_BPR_Cython import SLIM_BPR_Cython
from cf.item_cf import ItemBasedCollaborativeFiltering
from cf.user_cf import UserBasedCollaborativeFiltering
from cbf import ContentBasedFiltering
from MF.SVD_ICM import SVDRec
from SlimElasticNet import SLIMElasticNetRecommender
import numpy as np

from evaluation.evaluator import Evaluator
from lib.helper import Helper
from MF.ALS import AlternatingLeastSquare
from run import Runner


class HybridRecommender(object):
    def __init__(self, weights_seq, weights_short, weights_long, user_cf_param, item_cf_param, cbf_param,
                 slim_param, svd_param, ALS_param):


        self.tracks_df = pd.read_csv("../data/tracks.csv")

        ########## SET WEEIGHTS ##################
        self.w_seq = weights_seq
        self.w_short = weights_short
        self.w_long = weights_long

        ########### INIT ALGORITHMS ##############

        ###### USER CF #####
        self.userCF = UserBasedCollaborativeFiltering(knn=user_cf_param["knn"], shrink=user_cf_param["shrink"])
        self.userCF_sequential = UserBasedCollaborativeFiltering(knn=user_cf_param["knn"], shrink=user_cf_param["shrink"], use_tail_boost=True)

        ###### ITEM_CF #####
        self.itemCF = ItemBasedCollaborativeFiltering(knn=item_cf_param["knn"], shrink=item_cf_param["shrink"])
        self.itemCF_sequential = ItemBasedCollaborativeFiltering(knn=item_cf_param["knn"], shrink=item_cf_param["shrink"], use_tail_boost=True)

        ####### CBF #######
        self.cbf = ContentBasedFiltering(knn_alb=cbf_param["album_knn"], knn_art=cbf_param["artist_knn"],
                                         shrink_alb=cbf_param["album_shrink"], shrink_art=cbf_param["artist_shrink"])
        self.cbf_sequential = ContentBasedFiltering(knn_alb=cbf_param["album_knn"], knn_art=cbf_param["artist_knn"],
                                         shrink_alb=cbf_param["album_shrink"], shrink_art=cbf_param["artist_shrink"],
                                                    weight_alb=cbf_param["album_weight"], use_tail_boost=True)

        ###### SLIM ###### to edit more params just pass them to the init method (i.e. learning rate, batch_size)
        self.slim_sequential = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"], use_tail_boost=True)
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"], use_tail_boost=False)
        self.slim_elastic = SLIMElasticNetRecommender()
        self.slim_elastic_sequential = SLIMElasticNetRecommender(use_tail_boost=True)


        ##### SVD BASED ON ITEM CONTENT MATRIX #######
        ##### It takes too long to be computed and the increase in quality recommandations is quite low or none
        # self.svd_icm = SVDRec(n_factors=svd_param["n_factors"], knn=svd_param["knn"])
        # self.svd_icm_sequential = SVDRec(n_factors=svd_param["n_factors"], knn=svd_param["knn"], use_tail_boost=True)

        ###### ALS ######
        self.ALS = AlternatingLeastSquare(n_factors=ALS_param["n_factors"], regularization=ALS_param["reg"],
                                          iterations=ALS_param["iterations"])
        self.ALS_sequential = AlternatingLeastSquare(n_factors=ALS_param["n_factors"], regularization=ALS_param["reg"],
                                          iterations=ALS_param["iterations"], use_tail_boost=True)

    def fit(self, URM):
        self.URM = URM

        ### SUB-FITTING ###
        print("Fitting user cf...")
        self.userCF.fit(URM.copy())
        self.userCF_sequential.fit(URM.copy())

        print("Fitting item cf...")
        self.itemCF.fit(URM.copy())
        self.itemCF_sequential.fit(URM.copy())

        print("Fitting cbf...")
        self.cbf.fit(URM.copy())
        self.cbf_sequential.fit(URM.copy())

        print("Fitting slim...")
        self.slim_sequential.fit(URM.copy())
        self.slim_random.fit(URM.copy())

        print("Fitting slim elastic net...")
        self.slim_elastic.fit(URM.copy())
        self.slim_elastic_sequential.fit(URM.copy())

        # self.svd_icm.fit(URM.copy())
        # self.svd_icm_sequential.fit(URM.copy())

        print("Fitting ALS")
        self.ALS.fit(URM.copy())
        self.ALS_sequential.fit(URM.copy())


    def recommend(self, playlist_id, at=10):
        playlist_id = int(playlist_id)
        helper = Helper()

        ### DUE TO TIME CONSTRAINT THE CODE STRUCTURE HERE IS REDUNTANT
        ### TODO exploit inheritance to reduce code duplications and simple extract ratings, combine them, simply by iterate over a list of recommenders


        ### COMMON CODE ###
        self.hybrid_ratings = None #BE CAREFUL, MAGIC INSIDE :)


        ### COMBINE RATINGS IN DIFFERENT WAYS (seq, random short, random long)
        if(helper.is_sequential(playlist_id)):
            self.userCF_ratings = self.userCF_sequential.get_expected_ratings(playlist_id)
            self.itemCF_ratings = self.itemCF_sequential.get_expected_ratings(playlist_id)
            self.cbf_ratings = self.cbf_sequential.get_expected_ratings(playlist_id)
            self.slim_elastic_ratings = self.slim_elastic_sequential.get_expected_ratings(playlist_id)
            # self.svd_icm_ratings = self.svd_icm_sequential.get_expected_ratings(playlist_id)
            self.ALS_ratings = self.ALS_sequential.get_expected_ratings(playlist_id)
            self.slim_ratings = self.slim_sequential.get_expected_ratings(playlist_id)
            w_right = self.w_seq
        else:
            self.userCF_ratings = self.userCF.get_expected_ratings(playlist_id)
            self.itemCF_ratings = self.itemCF.get_expected_ratings(playlist_id)
            self.cbf_ratings = self.cbf.get_expected_ratings(playlist_id)
            self.slim_elastic_ratings = self.slim_elastic.get_expected_ratings(playlist_id)
            # self.svd_icm_ratings = self.svd_icm.get_expected_ratings(playlist_id)
            self.ALS_ratings = self.ALS.get_expected_ratings(playlist_id)
            self.slim_ratings = self.slim_random.get_expected_ratings(playlist_id)
            if len(self.URM[playlist_id].indices) > 10:
                w_right = self.w_long
            else:
                w_right = self.w_short

        self.hybrid_ratings = self.userCF_ratings * w_right["user_cf"]
        self.hybrid_ratings += self.itemCF_ratings * w_right["item_cf"]
        self.hybrid_ratings += self.cbf_ratings * w_right["cbf"]
        self.hybrid_ratings += self.slim_ratings * w_right["slim"]
        # self.hybrid_ratings += self.svd_icm_ratings * w_right["svd_icm"]
        self.hybrid_ratings += self.ALS_ratings * w_right["als"]
        self.hybrid_ratings += self.slim_elastic_ratings * w_right["elastic"]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.URM[playlist_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]



################################ PARAMETERS #################################
#################### Value for best sumbissions, MAP@10 on Kaggle = 0.1 ###########################

weights_short = {
    "user_cf": 0.03,
    "item_cf": 0.25,
    "cbf": 0.15,
    "icm_svd": 0,
    "als": 0.3,
    "slim": 0.6,
    "elastic": 1.5
}

weights_long = {
    "user_cf": 0.03,
    "item_cf": 0.35,
    "cbf": 0.2,
    "icm_svd": 0,
    "als": 0.3,
    "slim": 0.22,
    "elastic": 1.5
}

weights_seq = {
    "user_cf": 0.5,
    "item_cf": 0.1,
    "cbf": 0.72,
    "icm_svd": 0,
    "als": 0.14,
    "slim": 2.065,
    "elastic": 0.07,
}

user_cf_param = {
    "knn": 140,
    "shrink": 0
}

item_cf_param = {
    "knn": 310,
    "shrink": 0
}

cbf_param = {
    "album_knn": 45,
    "album_shrink": 8,
    "artist_knn": 25,
    "artist_shrink": 0,
    "album_weight": 0.85
}

slim_param = {
    "epochs": 40,
    "topK": 200
}

svd_param = {
    "n_factors": 2000,
    "knn": 100
}

ALS_param = {
    "n_factors": 300,
    "reg": 0.15,
    "iterations": 30
}

recommender = HybridRecommender(weights_seq=weights_seq, weights_long= weights_long, weights_short=weights_short, user_cf_param=user_cf_param,
                                item_cf_param=item_cf_param, cbf_param=cbf_param, slim_param=slim_param, svd_param=svd_param,
                                ALS_param=ALS_param)
Runner.run(is_test=True, recommender=recommender, split_type=None)
