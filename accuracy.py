"""
    computing accuracy metrics, and the index as follow:

    rmse
    mse
    mae
    fcp
"""

import numpy as np


def rmse(predictions, verbose):
    """compute RMSE(Root Mean Squared Error)."""
    if not predictions:
        raise ValueError("Prediction list is empty.")

    mse = np.mean([(true_r - est)**2
                  for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0: 1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose):
    """Compute MSE(Mean Squared Error)."""
    if not predictions:
        raise ValueError("Prediction list is empty.")

    mse_ = np.mean([(true_r - est)**2
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MSE: {0: 1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose):
    """Compute MAE(Mean Absolute Error)."""
    if not predictions:
        raise ValueError("Prediction list is empty.")

    mae_ = np.mean([abs(true_r - est)
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE: {0: 1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose):
    pass
