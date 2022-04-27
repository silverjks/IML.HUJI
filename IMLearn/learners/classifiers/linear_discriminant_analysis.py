from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)

        # calculate mu
        for k in self.classes_:
            n_k = np.count_nonzero(y == k)
            mu_k = np.sum(X[y == k], axis=0) / n_k
            if self.mu_ is None:
                self.mu_ = [mu_k]
            else:
                self.mu_ = np.concatenate([self.mu_, [mu_k]], axis=0)

        # calculate cov
        m = y.shape[0]
        for i, x_i in enumerate(X):
            if self.cov_ is None:
                self.cov_ = np.outer((x_i - self.mu_[y[i]]), (x_i - self.mu_[y[i]]))
            else:
                self.cov_ += np.outer((x_i - self.mu_[y[i]]), (x_i - self.mu_[y[i]]))

        self.cov_ = self.cov_ / (m - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)

        # calculate pi
        self.pi_ = np.bincount(y) / m

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        def normal_dist(mu, cov, X):
            det_cov = np.linalg.det(cov)
            dist = X - mu

            return np.power(np.e, -0.5 * dist @ self._cov_inv @ dist) /\
                   (np.sqrt(np.power(2 * np.pi, X.shape[0]) * det_cov))

        ll = []
        for x_i in X:
            x_i_ll = []
            for i in range(len(self.classes_)):
                x_i_ll.append(normal_dist(self.mu_[i], self.cov_, x_i) * self.pi_[i])
            ll.append(x_i_ll)
        return np.array(ll)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)
