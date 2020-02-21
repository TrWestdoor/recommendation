import numpy as np

from itertools import chain


def get_cv(cv):

    if cv is None:
        return KFolds(n_splits=5)
    else:
        return KFolds(n_splits=cv)


class KFolds:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, data):
        indices = np.arange(len(data.raw_ratings))

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits

            # assign remain items to first few fold.
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [data.raw_ratings[i] for i in chain(indices[:start], indices[stop:])]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_trainset(raw_testset)

            yield trainset, testset
