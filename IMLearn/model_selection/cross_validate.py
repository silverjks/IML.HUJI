from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_losses = np.empty(cv)
    validation_losses = np.empty(cv)

    folds = np.array_split(np.arange(X.shape[0]), cv)

    for i, indexes in enumerate(folds):
        # setup mask
        mask = np.ones_like(np.arange(X.shape[0]), bool)
        mask[indexes] = False

        # fit estimator
        estimator.fit(X[mask], y[mask])

        # make the predictions and calculate the losses
        train_prediction = estimator.predict(X[mask])
        train_losses[i] = scoring(train_prediction, y[mask])

        validation_prediction = estimator.predict(X[~mask])
        validation_losses[i] = scoring(validation_prediction, y[~mask])

    # return average train score over folds
    avg_train_score = np.mean(train_losses)

    # return average validation score over folds
    avg_validation_score = np.mean(train_losses)
    return avg_train_score, avg_validation_score
