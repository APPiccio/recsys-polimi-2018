from collections import defaultdict

import numpy as np
import pandas as pd
from random import randint, random, uniform
import time

from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from lib.helper import Helper

from lib.target_analyzer import TargetAnalyzer


class Evaluator(object):

    def __init__(self):
        self.URM_train = None
        self.URM_test = None
        self.dict_test = None
        self.target_playlists = None
        self.train_test_split = 0.80
        self.at = 10
        self.tracks_df = pd.read_csv("../data/tracks.csv")

        self.starting_weight_low= np.array([0.22, 0.14, 0.215, 0.45])

    def get_URM_train(self):
        return self.URM_train

    def get_dict_test(self):
        return self.dict_test

    def get_target_playlists(self):
        return self.target_playlists

    def get_URM_test(self):
        return self.URM_test

    #Split totally randomic, no cluster
    def split_randomic(self, URM, URM_df):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])
        n_target = len(available_playlists)*(1-self.train_test_split) #respecting the submission file we have to post
        while n_target>0:
            random_index = randint(0 , len(available_playlists) -1 )

            #chosen playlist_id
            playlist_id = available_playlists[random_index]
            tracks_left = len(URM[playlist_id].data)
            if tracks_left > 7:
                available_playlists = np.delete(available_playlists, np.where(available_playlists == playlist_id))
                selected_playlists = np.append(selected_playlists, playlist_id)
                n_target-=1

        self.target_playlists = selected_playlists.astype(int)

        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)
        count = 0

        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks
            count += 1

        all_tracks = self.tracks_df["track_id"].unique()

        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    # Split considering all playlists EXCEPT TARGET (because they will be surely afflicted by Kaggle test split)
    def split_randomic_all_playlists(self, URM, URM_df):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])

        helper = Helper()
        target_playlists_kaggle = helper.get_target_playlists_list()
        for playlist_id in available_playlists:
            if playlist_id not in target_playlists_kaggle:
                selected_playlists = np.append(selected_playlists, playlist_id)

        self.target_playlists = selected_playlists.astype(int)

        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)

        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()

        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items


    def split_randomic_all_playlists_longer(self, URM, URM_df, threshold_length=10):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])

        helper = Helper()
        target_playlists_kaggle = helper.get_target_playlists_list()
        for playlist_id in available_playlists:
            if playlist_id not in target_playlists_kaggle and len(URM[playlist_id].indices)>threshold_length:
                selected_playlists = np.append(selected_playlists, playlist_id)

        self.target_playlists = selected_playlists.astype(int)

        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)

        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()

        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    def split_randomic_all_playlists_longer_10000(self, URM, URM_df, threshold_length=10):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])

        helper = Helper()
        target_playlists_kaggle = helper.get_target_playlists_list()
        for playlist_id in available_playlists:
            if len(selected_playlists) == 10000:
                break
            if playlist_id not in target_playlists_kaggle and len(URM[playlist_id].indices)>threshold_length:
                selected_playlists = np.append(selected_playlists, playlist_id)

        self.target_playlists = selected_playlists.astype(int)

        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)

        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()

        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    def split_randomic_all_playlists_shorter(self, URM, URM_df, threshold_length = 10):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])

        helper = Helper()
        target_playlists_kaggle = helper.get_target_playlists_list()
        for playlist_id in available_playlists:
            if playlist_id not in target_playlists_kaggle and len(URM[playlist_id].indices)<=threshold_length:
                selected_playlists = np.append(selected_playlists, playlist_id)

        self.target_playlists = selected_playlists.astype(int)

        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)

        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()

        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    def split_randomic_exactly_last(self, URM, URM_df):
        # splitting URM in test set e train set
        selected_playlists = np.array([])

        helper = Helper()


        self.target_playlists = helper.get_target_playlists_list()[5000:]
        selected_playlists = self.target_playlists
        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)
        count = 0
        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])
            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks
            count += 1

        all_tracks = self.tracks_df["track_id"].unique()

        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    #Split getting 5000 sequentially, 5000 randomic with cluster segment = 1
    def split_sequential(self, URM, URM_df):
        segment = 1
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])
        target_analyzer = TargetAnalyzer()

        #Gets distribution of only last 5000 playlists
        dist = target_analyzer.get_distribution_array_only_last(segment)

        helper = Helper()
        target_playlists = helper.get_target_playlists_list()[:5000]
        #n_target = np.sum(dist) - len(target_playlists)

        # Removing from the cluster distribution the len of the sequential target
        for playlist_id in target_playlists:
            playlist_id = int(playlist_id)
            available_playlists = np.delete(available_playlists, np.where(available_playlists == playlist_id))
            selected_playlists = np.append(selected_playlists, playlist_id)
            #target_len = len(URM[playlist_id].data)
            #dist[target_len] -= 1

        print("Clustering with segment = " + str(segment))
        for key in tqdm(range(len(dist))):
            while dist[key]!=0:
                random_index = randint(0, len(available_playlists) - 1)
                playlist_id = available_playlists[random_index]
                target_segment = int(0.8*len(URM[playlist_id].data))
                if target_segment==key:
                    available_playlists = np.delete(available_playlists, np.where(available_playlists == playlist_id))
                    selected_playlists = np.append(selected_playlists, playlist_id)
                    dist[key]-=1

        self.target_playlists = selected_playlists.astype(int)
        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)

        for playlist_id in selected_playlists:
            #Tracks = lista delle tracks prese dalla URM
            tracks = np.array(grouped[playlist_id])
            if playlist_id in target_playlists:
                to_be_removed = int(len(tracks) * 0.2)

                #Torna le #to_be_removed tracks ordinate sequenzialmente. e le toglie dalla lista delle tracks
                to_be_removed_tracks = helper.get_sorted_tracks_in_playlist(playlist_id)[-to_be_removed:]
                for track in to_be_removed_tracks:
                    relevant_items[playlist_id].append(track)
                    tracks = np.delete(tracks, np.where(tracks == track))
            else:
                to_be_removed = int(len(tracks) * 0.2)
                for i in range(to_be_removed):
                    index = randint(0, len(tracks) - 1)
                    removed_track = tracks[index]
                    relevant_items[playlist_id].append(removed_track)
                    tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()
        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    def split_only_sequential(self, URM, URM_df):

        helper = Helper()

        sequential_playlists = helper.get_target_playlists_list()[:5000]
        selected_playlists = np.array([])


        self.target_playlists = sequential_playlists

        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)
        for playlist_id in sequential_playlists:

            # Tracks = lista delle tracks prese dalla URM
            tracks = np.array(grouped[playlist_id])
            to_be_removed = int(len(tracks) * 0.2)

            # Torna le #to_be_removed tracks ordinate sequenzialmente. e le toglie dalla lista delle tracks
            to_be_removed_tracks = helper.get_sorted_tracks_in_playlist(playlist_id)[-to_be_removed:]
            for track in to_be_removed_tracks:
                relevant_items[playlist_id].append(track)
                tracks = np.delete(tracks, np.where(tracks == track))
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()
        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items

    # OLD_SPLIT (Piccio)
    def split(self, URM, URM_df):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])
        target_analyzer = TargetAnalyzer()
        segment_size = 2
        min_playlist_len_after_split = 5
        dist = target_analyzer.get_distribution_array(segment_size=segment_size)
        # in this way n_target = 10000
        helper = Helper()
        target_playlists = helper.get_target_playlists_list()[:5000]
        n_target = np.sum(dist)  # - len(target_playlists)


        while n_target > 0:
            random_index = randint(0, len(available_playlists) - 1)
            playlist_id = available_playlists[random_index]
            target_len = len(URM[playlist_id].data) * 0.8
            if target_len > min_playlist_len_after_split:
                target_segment = int(target_len / segment_size)
                while dist[target_segment] <= 0:
                    random_index = randint(0, len(available_playlists) - 1)
                    playlist_id = available_playlists[random_index]
                    target_len = len(URM[playlist_id].data) * 0.8
                    if target_len > min_playlist_len_after_split:
                        target_segment = int(target_len / segment_size)
                n_target -= 1
                dist[target_segment] -= 1
                selected_playlists = np.append(selected_playlists, playlist_id)
                available_playlists = np.delete(available_playlists, np.where(available_playlists == playlist_id))

        self.target_playlists = selected_playlists.astype(int)
        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        grouped_test = grouped.copy()

        relevant_items = defaultdict(list)
        count = 0
        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks
            grouped_test[playlist_id] = relevant_items[playlist_id]
            count += 1
        all_tracks = self.tracks_df["track_id"].unique()
        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.dict_test = relevant_items
        self.URM_test = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped_test)
        self.URM_test = self.URM_test.tocsr()
        self.URM_test = self.URM_test.astype(np.float64)
        self.URM_train = self.URM_train.astype(np.float64)

    #Get all randomic playlists applying cluster, segment = 2
    def split_cluster_randomic(self, URM, URM_df):
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])
        target_analyzer = TargetAnalyzer()
        segment_size = 1
        min_playlist_len_after_split = 5
        dist = target_analyzer.get_distribution_array(segment_size=segment_size)
        # in this way n_target = 10000
        helper = Helper()
        target_playlists = helper.get_target_playlists_list()[:5000]
        n_target = np.sum(dist)  # - len(target_playlists)

        while n_target > 0:
            random_index = randint(0, len(available_playlists) - 1)
            playlist_id = available_playlists[random_index]
            target_len = len(URM[playlist_id].data) * 0.8
            if target_len > min_playlist_len_after_split:
                target_segment = int(target_len / segment_size)
                while dist[target_segment] <= 0:
                    random_index = randint(0, len(available_playlists) - 1)
                    playlist_id = available_playlists[random_index]
                    target_len = len(URM[playlist_id].data) * 0.8
                    if target_len > min_playlist_len_after_split:
                        target_segment = int(target_len / segment_size)
                n_target -= 1
                dist[target_segment] -= 1
                selected_playlists = np.append(selected_playlists, playlist_id)
                available_playlists = np.delete(available_playlists, np.where(available_playlists == playlist_id))

        self.target_playlists = selected_playlists.astype(int)
        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        grouped_test = grouped.copy()

        relevant_items = defaultdict(list)
        count = 0
        for playlist_id in selected_playlists:
            tracks = np.array(grouped[playlist_id])
            # if playlist_id in target_playlists:
            # to_be_removed = int(len(tracks) * 0.2)
            # to_be_removed_tracks = helper.get_sorted_tracks_in_playlist(playlist_id)[-to_be_removed:]
            # for track in to_be_removed_tracks:
            #     relevant_items[playlist_id].append(track)
            #     tracks = np.delete(tracks, np.where(tracks == track))
            # for i in range(to_be_removed):
            #    removed_track = tracks[-1]
            #    relevant_items[playlist_id].append(removed_track)
            #    tracks = np.delete(tracks, len(tracks) - 1)
            # else:
            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks
            grouped_test[playlist_id] = relevant_items[playlist_id]
            count += 1
        all_tracks = self.tracks_df["track_id"].unique()
        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.dict_test = relevant_items
        # bib URM
        # self.URM_train = helper.get_urm_csr_bib(URM = self.URM_train)
        # plotter = TargetAnalyzer()
        # plotter.plot_standard_distribution()
        # plotter.plot_distribution(self.URM_train, self.target_playlists)
        self.URM_test = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped_test)
        self.URM_test = self.URM_test.tocsr()
        self.URM_test = self.URM_test.astype(np.float64)
        self.URM_train = self.URM_train.astype(np.float64)

        # Get all randomic playlists of LAST 5000 targret playlists applying cluster, segment = 1

    def split_cluster_randomic_only_last(self, URM, URM_df):
        # splitting URM in test set e train set
        segment = 1
        # splitting URM in test set e train set
        selected_playlists = np.array([])
        available_playlists = np.arange(URM.shape[0])
        target_analyzer = TargetAnalyzer()

        # Gets distribution of only last 5000 playlists
        dist = target_analyzer.get_distribution_array_only_last(segment)

        helper = Helper()
        target_playlists = helper.get_target_playlists_list()[:5000] # WILL REMOVE THEM

        print("Clustering with segment = " + str(segment))
        for key in tqdm(range(len(dist))):
            while dist[key] != 0:
                random_index = randint(0, len(available_playlists) - 1)
                playlist_id = available_playlists[random_index]
                target_segment = int(0.8 * len(URM[playlist_id].data))
                if target_segment == key and playlist_id not in target_playlists:
                    available_playlists = np.delete(available_playlists, np.where(available_playlists == playlist_id))
                    selected_playlists = np.append(selected_playlists, playlist_id)
                    dist[key] -= 1

        self.target_playlists = selected_playlists.astype(int)
        grouped = URM_df.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        relevant_items = defaultdict(list)

        for playlist_id in selected_playlists:
            # Tracks = lista delle tracks prese dalla URM
            tracks = np.array(grouped[playlist_id])

            to_be_removed = int(len(tracks) * 0.2)
            for i in range(to_be_removed):
                index = randint(0, len(tracks) - 1)
                removed_track = tracks[index]
                relevant_items[playlist_id].append(removed_track)
                tracks = np.delete(tracks, index)
            grouped[playlist_id] = tracks

        all_tracks = self.tracks_df["track_id"].unique()
        matrix = MultiLabelBinarizer(classes=all_tracks, sparse_output=True).fit_transform(grouped)
        self.URM_train = matrix.tocsr()
        self.URM_train = self.URM_train.astype(np.float64)
        self.dict_test = relevant_items



    def MAP(self, recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(len(is_relevant)))
        map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])
        return map_score

    def evaluate(self, playlist_id, recommended_items):
        relevant_items = self.dict_test[playlist_id]
        map = self.MAP(recommended_items, relevant_items)
        return map

    #evaluates just weights assigned in the run method. No array weights neede
    def global_evaluate_single(self, recommender):
        MAP_final = 0
        recommender.fit(self.URM_train)
        count = 0
        for target in tqdm(self.target_playlists):
            recommended_items = recommender.recommend(target)
            MAP_final += self.evaluate(target, recommended_items)
            count +=1
        MAP_final /= len(self.target_playlists)
        return MAP_final
