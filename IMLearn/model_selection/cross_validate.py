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
    f_X = np.array_split(X, cv)
    f_y = np.array_split(y, cv)

    train_losses = np.zeros(cv)
    validation_losses = np.zeros(cv)

    for i in range(cv):
        X_train = np.concatenate(f_X[:i] + f_X[i + 1:], axis=0)
        y_train = np.concatenate(f_y[:i] + f_y[i + 1:], axis=0)

        X_pred = f_X[i]
        y_pred = f_y[i]

        estimator.fit(X_train, y_train)

        train_losses[i] = scoring(estimator.predict(X_train), y_train)
        validation_losses[i] = scoring(estimator.predict(X_pred), y_pred)

    return np.average(train_losses), np.average(validation_losses)
