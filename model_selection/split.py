import numpy as np

from itertools import chain
from surprise.utils import get_rng


def get_cv(cv):

    if cv is None:
        return KFold(n_splits=5)
    else:
        return KFold(n_splits=cv)


class KFold():
    def __init__(self, n_splits=5, random_state=None, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, data):
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

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
            testset = data.construct_testset(raw_testset)

            yield trainset, testset
