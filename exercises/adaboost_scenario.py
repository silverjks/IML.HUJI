import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    booster = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    training_error = []
    for t in range(1, n_learners + 1):
        error = booster.partial_loss(train_X, train_y, t)
        training_error.append(error)

    test_error = []

    # for Q3:
    min_error = np.inf
    min_error_t = 0

    for t in range(1, n_learners + 1):
        error = booster.partial_loss(test_X, test_y, t)
        test_error.append(error)
        # for Q3
        if error < min_error:
            min_error = error
            min_error_t = t

    figure = go.Figure([
        go.Scatter(x=np.arange(n_learners), y=np.array(test_error)),
        go.Scatter(x=np.arange(n_learners), y=np.array(training_error))
    ])
    figure.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    subplot_titles = [f"num classifiers = {t}" for t in T]

    figure = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)

    for i, t in enumerate(T):
        figure.add_traces([decision_surface(lambda x: booster.partial_predict(x, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    figure.update_layout(margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    figure.show()

    # Question 3: Decision surface of best performing ensemble

    min_accuracy = accuracy(test_y, booster.partial_predict(test_X, min_error_t))
    subplot_titles = [f"num classifiers = {min_error_t} ; accuracy = {min_accuracy}"]
    figure = make_subplots(rows=1, cols=1, subplot_titles=subplot_titles)

    for i, t in enumerate(T):
        figure.add_traces([decision_surface(lambda x: booster.partial_predict(x, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))], rows=1, cols=1)

    figure.update_layout(margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    figure.show()

    # Question 4: Decision surface with weighted samples
    sample_sizes = booster.D_ / np.max(booster.D_) * 5

    figure = go.Figure([decision_surface(booster.predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=train_y, size=sample_sizes, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))])
    figure.update_layout(title=f"proportional weight of training set after last iteration")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.show()

x
if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
