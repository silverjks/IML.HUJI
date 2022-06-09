from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise

    # STEP 1: set up f, as defined above
    f = np.vectorize(lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2))

    # STEP 2: generate samples
    range_start = -1.2
    range_fin = 2
    X = np.linspace(start=range_start, stop=range_fin, num=n_samples)

    # STEP 3: generate noise (according to the model y-f(x)+epsilon, as required)
    epsilon = np.random.normal(loc=0, scale=noise, size=n_samples)
    y = f(X) + epsilon

    # and split into training- and testing portions
    train_portion = 2 / 3
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_portion)

    # we need x_train and x_test to be Series for np.vander in Q2
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    # scatter plot the true (noiseless) ,odel and the two sets using different colors for train and test samples.
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=X, y=f(X), name="noiseless", mode="markers"))
    figure.add_trace(go.Scatter(x=x_train, y=y_train, name="train", mode="markers"))
    figure.add_trace(go.Scatter(x=x_test, y=y_test, name="set", mode="markers"))
    figure.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    num_of_degrees = 11  # cause 0->10
    train_scores = []
    validation_scores = []

    for deg in range(num_of_degrees):
        # fit model
        model = PolynomialFitting(deg)

        # default in cross_validate is 5-fold
        train_res, validation_res = cross_validate(model, x_train, y_train, mean_square_error)

        train_scores.append(train_res)
        validation_scores.append(validation_res)

    x = np.arange(num_of_degrees)
    figure = go.Figure(data=[go.Scatter(x=x, y=train_scores, name="Train Error", mode="lines"),
                             go.Scatter(x=x, y=validation_scores, name="Validation Error", mode="lines")])
    figure.update_layout(xaxis_title="Degree", yaxis_title="Loss")
    figure.show()

    # Question 3 - Using best value of k
    k = np.argmin(validation_scores)

    #  fit a k-degree polynomial model and report test error
    model = PolynomialFitting(k).fit(x_train, y_train)

    # round to 2 decimal places
    loss = np.round(model.loss(x_test, y_test), 2)
    validation = np.round(validation_scores[k], 2)
    print("The test error for the best k, which is", k, ", with noise", noise, ", is", loss,
          ". previous validation error is ", validation)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    x_train, x_test, y_train, y_test = X[:n_samples], X[n_samples:], y[:n_samples], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_scores = []
    ridge_validation_scores = []
    lasso_train_scores = []
    lasso_validation_scores = []

    range_start = 0
    range_fin = 2
    possible_range = np.linspace(range_start, range_fin, num=n_evaluations)

    for i, l in enumerate(possible_range):
        # ridge
        ridge_model = RidgeRegression(l)
        train_res, validation_res = cross_validate(ridge_model, x_train, y_train, mean_square_error)
        ridge_train_scores.append(train_res)
        ridge_validation_scores.append(validation_res)

        # lasso
        lasso_model = Lasso(l)
        train_res, validation_res = cross_validate(lasso_model, x_train, y_train, mean_square_error)
        lasso_train_scores.append(train_res)
        lasso_validation_scores.append(validation_res)

    figure = go.Figure(
        data=[go.Scatter(x=possible_range, y=ridge_train_scores, name="ridge: avg train error", mode="lines"),
              go.Scatter(x=possible_range, y=ridge_validation_scores, name="ridge: avg validation error", mode="lines"),
              go.Scatter(x=possible_range, y=lasso_train_scores, name="lasso: avg train error", mode="lines"),
              go.Scatter(x=possible_range, y=lasso_validation_scores, name="lasso: avg train error", mode="lines")])
    figure.update_layout(xaxis_title="Lambda", yaxis_title="Loss")
    figure.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    print("model: ridge")
    ridge_l = possible_range[np.argmin(ridge_validation_scores)]
    print("best lambda is", ridge_l)
    ridge_loss = RidgeRegression(ridge_l).fit(x_train, y_train).loss(x_test, y_test)
    print("loss is", ridge_loss)

    print("model: lasso")
    lasso_l = possible_range[np.argmin(lasso_validation_scores)]
    print("best lambda is", lasso_l)
    lasso_loss = mean_square_error(Lasso(lasso_l).fit(x_train, y_train).predict(x_test), y_test)
    print("loss is", lasso_loss)

    least_squares_loss = LinearRegression().fit(x_train, y_train).loss(x_test, y_test)
    print("model: least squares")
    print("loss:", least_squares_loss)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
