import numpy as np
import matplotlib.pyplot as plt

from helper import Helper


class TargetAnalyzer(object):

    def plot_distribution(self, URM, target_playlists):
        helper = Helper()
        max = helper.get_max_min_len_line(URM)[1]
        target_len = np.zeros(max)
        for playlist in target_playlists:
            playlist = int(playlist)
            this_len = len(URM[playlist].data)
            target_len[this_len] += 1
        plt.hist(target_len, color='blue', edgecolor='black',
                 bins=int(180 / 5))
        plt.show()

    def plot_standard_distribution(self):
        target_file = open("../data/target_playlists.csv", "r")
        target_playlists = list(target_file)[1:]
        helper = Helper()
        URM = helper.get_urm_csr()
        self.plot_distribution(URM, target_playlists)

    def get_distribution_array(self, segment_size=5):
        helper = Helper()
        URM = helper.get_urm_csr()
        target_file = open("../data/target_playlists.csv", "r")
        target_playlists = list(target_file)[1:]
        max = helper.get_max_min_len_line(URM)[1]
        dist = np.zeros(int(round(max/segment_size, 0)) + 1)
        for playlist in target_playlists[5000:]:
            playlist = int(playlist)
            seg_id = int(len(URM[playlist].data) / segment_size)
            dist[seg_id] += 1
        return dist


    # Get only last 5000 array distribution, segment always = 1
    def get_distribution_array_only_last(self, segment_size=1):
        helper = Helper()
        URM = helper.get_urm_csr()
        target_file = open("../data/target_playlists.csv", "r")
        target_playlists = list(target_file)[5001:]
        max = helper.get_max_min_len_line(URM)[1]
        dist = np.zeros(max + 1)
        for playlist in target_playlists:
            playlist = int(playlist)
            seg_id = len(URM[playlist].data)
            dist[seg_id] += 1
        return dist

    def get_custom_distribution_array(self, URM, target_playlists, segment_size=5):
        helper = Helper()
        max = helper.get_max_min_len_line(URM)[1]
        dist = np.zeros(round(max/segment_size) + 1)
        for playlist in target_playlists:
            playlist = int(playlist)
            seg_id = int(len(URM[playlist].data) / segment_size)
            dist[seg_id] += 1
        return dist

    def get_distribution_diff(self, dist1, dist2):
        dist = np.zeros(len(dist1))
        for i in range(len(dist1)):
            dist[i] = dist1[i] - dist2[i]
        return dist

    def sorted_playlists_distribution(self, segment_size=5):
        helper = Helper()
        URM = helper.get_urm_csr()
        target_file = open("../data/target_playlists.csv", "r")
        target_playlists = list(target_file)[1:]
        max = helper.get_max_min_len_line(URM)[1]
        dist = np.zeros(int(round(max/segment_size, 0)) + 1)
        for playlist in target_playlists[:5000]:
            playlist = int(playlist)
            seg_id = int(len(URM[playlist].data) / segment_size)
            dist[seg_id] += 1
        return dist

