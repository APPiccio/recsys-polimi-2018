import pandas as pd
from tqdm import tqdm

from evaluator import Evaluator
from helper import Helper


class Runner(object):

    @staticmethod
    def run(is_test, recommender, split_type=None, threshold=10):
        target_file = open("../data/target_playlists.csv", 'r')
        URM_df = pd.read_csv("../data/train.csv")
        helper = Helper()
        URM = helper.get_urm_csr()


        if is_test:
            print("Starting testing phase..")
            evaluator = Evaluator()

            if split_type is None:
                evaluator.split(URM, URM_df)

            elif split_type == "randomic_all_playlists":
                evaluator.split_randomic_all_playlists(URM, URM_df)

            elif split_type == "split_randomic_all_playlists_longer":
                evaluator.split_randomic_all_playlists(URM, URM_df, threshold)

            elif split_type == "split_randomic_all_playlists_longer_10000":
                evaluator.split_randomic_all_playlists(URM, URM_df, threshold)

            elif split_type == "split_randomic_all_playlists_shorter":
                evaluator.split_randomic_all_playlists_shorter(URM, URM_df, threshold)

            elif split_type == "split_randomic_exactly_last":
                evaluator.split_randomic_exactly_last(URM, URM_df)

            elif split_type == "split_sequential":
                evaluator.split_sequential(URM, URM_df)

            elif split_type == "split_only_sequential":
                evaluator.split_only_sequential(URM, URM_df)

            elif split_type == "split_cluster_randomic":
                evaluator.split_cluster_randomic(URM, URM_df)

            elif split_type == "split_cluster_randomic_only_last":
                evaluator.split_cluster_randomic_only_last(URM, URM_df)

            score = evaluator.global_evaluate_single(recommender)
            print("Evalution completed, score = " + str(score) + "\n")
            return score

        else:

            print("Starting prediction to be submitted..")
            recommender.fit(URM)
            submission_file = open("../data/submission.csv", "w")
            target_playlists = list(target_file)[1:]
            submission_file.write("playlist_id,track_ids\n")
            for target in tqdm(target_playlists):
                res = recommender.recommend(target)
                res = " ".join(str(x) for x in res)
                target = target.replace("\n", "")
                submission_file.write(str(target) + "," + str(res) + "\n")
            submission_file.close()
            print("Saved predictions to file")

