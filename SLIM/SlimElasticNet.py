
import pathos.pools as pp
import multiprocessing
from functools import partial
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet

from helper import Helper
from run import Runner


class SLIMElasticNetRecommender(object):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """
    def __init__(self, alpha=1e-4, l1_ratio=0.1, fit_intercept=False, copy_X=False, precompute=False, selection='random',
                max_iter=100, tol=1e-4, topK=100, positive_only=True, workers=multiprocessing.cpu_count(), use_tail_boost=False):

        self.analyzed_items = 0
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.precompute = precompute
        self.selection = selection
        self.max_iter = max_iter
        self.tol = tol
        self.topK = topK
        self.positive_only = positive_only
        self.workers = workers
        self.use_tail_boost = use_tail_boost

    """ 
        Fit given to each pool thread, to fit the W_sparse 
    """
    def _partial_fit(self, currentItem, X):

        model = ElasticNet(alpha=self.alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=self.fit_intercept,
                           copy_X=self.copy_X,
                           precompute=self.precompute,
                           selection=self.selection,
                           max_iter=self.max_iter,
                           tol=self.tol)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)

        local_topK = min(len(model.sparse_coef_.data) - 1, self.topK)

        relevant_items_partition = (-model.sparse_coef_.data).argpartition(local_topK)[0:local_topK]
        relevant_items_partition_sorting = np.argsort(-model.sparse_coef_.data[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        non_zero_mask = model.coef_[ranking] > 0.0
        ranking = ranking[non_zero_mask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self, URM):

        self.URM_train = sps.csc_matrix(URM)

        if self.use_tail_boost:
            helper = Helper()
            self.URM = helper.tail_boost(self.URM)

        n_items = self.URM_train.shape[1]
        print("Iterating for " + str(n_items) + "times")

        #create a copy of the URM since each _pfit will modify it
        copy_urm = self.URM_train.copy()

        _pfit = partial(self._partial_fit, copy_urm)
        pool = pp.ProcessPool(self.workers)
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def get_expected_ratings(self, playlist_id):
        playlist_id = int(playlist_id)
        user_profile = self.URM_train[playlist_id]
        expected_ratings = user_profile.dot(self.W_sparse).toarray().ravel()

        # # EDIT
        return expected_ratings

    def recommend(self, playlist_id, at=10):
        playlist_id = int(playlist_id)
        # compute the scores using the dot product
        scores = self.get_expected_ratings(playlist_id)
        user_profile = self.URM_train[playlist_id].indices
        scores[user_profile] = 0

        # rank items
        recommended_items = np.flip(np.argsort(scores), 0)

        return recommended_items[:at]


# recommender = SLIMElasticNetRecommender()
# Runner.run(True, recommender, None)
