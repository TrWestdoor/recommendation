from .split import get_cv
from joblib import Parallel, delayed

import accuracy
import time
import numpy as np


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   return_train_measures=False, n_jobs=1,
                   pre_dispatch='2*n_jobs', verbose=False):
    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = (delayed(fit_and_score)(algo, trainset, testset, measures, return_train_measures)
                    for (trainset, testset) in cv.split(data))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_list)

    (test_measures_dicts,
     train_measures_dicts,
     fit_times,
     test_times) = zip(*out)

    test_measures = dict()
    train_measures = dict()
    ret = dict()
    for m in measures:
        # transformer tuple of dicts to dict of lists.
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
        ret['test_'+m] = test_measures[m]

        if return_train_measures:
            pass

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times

    if verbose:
        pass

    return ret


def fit_and_score(algo, trainset, testset, measures,
                  return_train_measures=False):
    """ Achieve train an algorithm and compute accuracy measure,
        also return train and test time.
    Args:
        algo: algorithm to use.
        trainset(:obj:`Trainset <surprise.trainset.Trainset>`): The trainset.
        testset: The testset.
        measures(list): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module.
        return_train_measures(bool): Whether to compute performance measures on
            the trainset. Default is ``False``.
    Returns:
        tuple: A tuple containing:
            - A dictionary mapping each accuracy metric to its value on the
            testset (keys are lower case).

            - A dictionary mapping each accuracy metric to its value on the
            trainset (keys are lower case). This dict is empty if
            return_train_measures is False.

            - The fit time in seconds.

            - The testing time in seconds.
    """

    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    if return_train_measures:
        pass

    test_measures = dict()
    train_measures = dict()
    for m in measures:
        f = getattr(accuracy, m.lower())
        test_measures[m] = f(predictions, verbose=0)

        if return_train_measures:
            pass

    return test_measures, train_measures, fit_time, test_time

