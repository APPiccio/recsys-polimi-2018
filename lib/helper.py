import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import sparse
from sklearn import feature_extraction
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from tqdm import tqdm

from IR_feature_weighting import okapi_BM_25


class Helper(object):
    def __init__(self):
        self.URM_df = pd.read_csv("../data/new_train.csv")
        self.playlists_list = np.asarray(list(self.URM_df.playlist_id))
        self.tracks_list = np.asarray(list(self.URM_df.track_id))

    def get_urm_csr(self):
        ratings_list = list(np.ones(len(self.tracks_list)))
        URM = sps.coo_matrix((ratings_list, (self.playlists_list, self.tracks_list)), dtype=np.float64)
        URM = URM.tocsr()
        return URM

    def get_sorted_tracks_in_playlist(self, playlist_id):
        index_list = np.where(self.playlists_list == playlist_id)
        return self.tracks_list[index_list]

    def wipe_row_csr(self, mat, row_index):
        mat.data[mat.indptr[row_index]:mat.indptr[row_index + 1]] = 0.0


    def get_max_min_len_line(self, URM):
        max = -np.inf
        min = np.inf
        for line in URM:
            count = len(line.data)
            if count > max:
                max = count
            if count < min:
                min = count
        return (min, max)

    def get_icm_artist(self):
        tracks_df = pd.read_csv("../data/tracks.csv")
        tracks_list = tracks_df.track_id
        artist_list = tracks_df.artist_id
        tracks_list = list(tracks_list)
        artist_list = list(artist_list)
        values_list = list(np.ones(len(artist_list)))
        ICM = sps.coo_matrix((values_list, (tracks_list, artist_list)), dtype=np.float64)
        #ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
        #ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')
        return ICM.tocsr()

    def get_icm_album(self):
        tracks_df = pd.read_csv("../data/tracks.csv")
        tracks_list = tracks_df.track_id
        tracks_list = list(tracks_list)
        album_list = tracks_df.album_id
        album_list = list(album_list)
        values_list = list(np.ones(len(album_list)))
        ICM = sps.coo_matrix((values_list, (tracks_list, album_list)), dtype=np.float64)
        #ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
        #ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')
        return ICM.tocsr()

    def get_target_playlists_list(self):
        target_file = open("../data/target_playlists.csv")
        t_list = list(target_file)[1:]
        t_list_int = list()
        for x in t_list:
            t_list_int.append(int(x))
        return t_list_int

    def get_icm(self):
        tracks_data = pd.read_csv("../data/tracks.csv")
        artists = tracks_data.reindex(columns=['track_id', 'artist_id'])
        artists.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
        artists_list = [[a] for a in artists['artist_id']]
        icm_artists = MultiLabelBinarizer(sparse_output=True).fit_transform(artists_list)
        icm_artists_csr = icm_artists.tocsr()

        albums = tracks_data.reindex(columns=['track_id', 'album_id'])
        albums.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
        albums_list = [[a] for a in albums['album_id']]
        icm_albums = MultiLabelBinarizer(sparse_output=True).fit_transform(albums_list)
        icm_albums_csr = icm_albums.tocsr()

        durations = tracks_data.reindex(columns=['track_id', 'duration_sec'])
        durations.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
        durations_list = [[d] for d in durations['duration_sec']]
        icm_durations = MultiLabelBinarizer(sparse_output=True).fit_transform(durations_list)
        icm_durations_csr = icm_durations.tocsr()

        ICM = sparse.hstack((icm_albums_csr, icm_artists_csr, icm_durations_csr))
        ICM_csr = ICM.tocsr()
        return ICM_csr

    def get_URM_BM_25(self, URM):
        return okapi_BM_25(URM)

    def get_URM_tfidf(self, URM):
        URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM)
        return URM_tfidf.tocsr()

    def get_mat_normalize(self, URM, axis=0):
        return normalize(URM, axis=axis, norm='l2').tocsr()

    def get_float_input(self, message):
        while True:
            user_number = input(message)
            try:
                val = float(user_number)
                if (val >= 0):
                    return val
            except ValueError:
                print("No.. input string is not a number. It's a string")


    def tail_boost(self, URM, step=1, lastN=4):
        target_playlists = self.get_target_playlists_list()
        for row_index in tqdm(range(URM.shape[0])):
            if row_index in target_playlists:
                sorted_tracks = self.get_sorted_tracks_in_playlist(row_index)
                tracks = URM[row_index].indices
                # sorted_tracks = np.intersect1d(sorted_tracks, tracks)
                lenTracks = len(tracks)

                for i in range(lenTracks):
                    # THE COMMA IS IMPORTANT
                    index_of_track, = np.where(sorted_tracks == tracks[i])
                    if lenTracks - index_of_track <= lastN:
                        additive_score = ((lastN+1) - (lenTracks - index_of_track)) * step
                        URM.data[URM.indptr[row_index] + i] += additive_score
        return URM


    def is_sequential(self, playlist_id):
        return playlist_id in self.get_target_playlists_list()[:5000]
