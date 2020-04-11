import heapq

from surprise.prediction_algorithms import PredictionImpossible
from .algo_base import AlgoBase


class SymmetricalAlgo(AlgoBase):
    def __init__(self, sim_options={}, verbose=True, **kwargs):
        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = trainset.n_users if ub else trainset.n_items
        self.n_y = trainset.n_items if ub else trainset.n_users
        self.xr = trainset.ur if ub else trainset.ir
        self.yr = trainset.ir if ub else trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class KNNBasic(SymmetricalAlgo):
    """
    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set to the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
    """
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):
        SymmetricalAlgo.__init__(self, sim_options=sim_options, verbose=verbose, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        SymmetricalAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r) for x2, r in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim*r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details
