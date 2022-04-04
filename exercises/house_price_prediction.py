import os
import random

from plotly.io import kaleido

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Loading the data
    X = pd.read_csv(filename)

    # Preprocessing

    # Step 1: remove empty rows.
    X.dropna(inplace=True)

    # Step 2: clean up the data

    # Step 2.1: using a helper func that can be found in the Answers.pdf file, I located outliers
    # (specifically, small values) in the columns that have a numerical value: id, bathrooms, floors, view, condition,
    # grade,sqft_above,sqft_basement,yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15.

    # not all small values are invalid!! those are, however:
    # invalid id, floors, condition, grade, sqft_above, yr_built, zipcode,sqft_living15 (specifically, 0): 20671
    # invalid sqft_lot15: 20671, 15298
    # invalid bathrooms (specifically, 0 bathrooms): 875, 1149, 3119, 5832, 6994, 9773, 9854, 10481, 14423, 19452, 20671
    # invalid price (non-positive): 6383, 15504, 20671

    X.drop([875, 1149, 3119, 5832, 6994, 9773, 9854, 10481, 14423, 19452, 20671, 6383, 15504, 15298], inplace=True)

    # Step 2.2: remove columns that I think are irrelevant:
    X.drop(["id"], inplace=True, axis=1)

    # Step 2.3.1: date breakdown:
    year_data = pd.to_numeric(X['date'].str.slice(start=0, stop=4))
    year_frame = year_data.to_frame(name='year')
    month_data = pd.to_numeric(X['date'].str.slice(start=4, stop=6))
    month_frame = month_data.to_frame(name='month')
    day = pd.to_numeric(X['date'].str.slice(start=6, stop=8))
    day_frame = day.to_frame(name='day')

    # Step 2.3.2: remove date from existing data:
    X.drop(["date"], inplace=True, axis=1)

    # Step 2.4: make zipcode numeric with one hot encoding
    zipcode_dummies = pd.get_dummies(X["zipcode"])

    # Step 2.5: add columns:
    X = pd.concat([X, year_frame, month_frame, day_frame, zipcode_dummies], axis=1)

    # Step 3.1: set y to price
    y = X.pop('price')

    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def pearson_correlation(x, y):
        return (np.cov(x, y)[0][1]) / (np.std(x) * np.std(y))

    features = ["year", "month", "day", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
                "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode",
                "lat", "long", "sqft_living15", "sqft_lot15"]
    for feature in features:
        column = X[feature]
        ro = pearson_correlation(column, y)
        figure = go.Figure([go.Scatter(x=column.to_numpy(), y=y, mode='markers', name=r'$$')],
                           layout=go.Layout(
                               title=f'Correlation Between the Feature \'{feature}\' and Response, with Pearson '
                                     f'Correlation {ro}', xaxis_title=f'{feature}',
                               yaxis_title=r"$\text{Prices}$", height=500))
        figure.show()
        # figure.write_image("" + output_path + "/" + feature + ".pdf")


if __name__ == '__main__':
    np.random.seed(0)
    filename = '/home/silver/IML.HUJI/datasets/house_prices.csv'
    # Question 1 - Load and preprocessing of housing prices dataset

    X, y = load_data(filename)



    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, 'images')

    # # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:

    training_sizes = []
    repetitions = 10
    mean_s = []
    lower_bound = []
    upper_bound = []

    for p in range(10, 100 + 1):
        training_sizes.append(p)
        p_losses = np.array([])
        for i in range(repetitions):
            #   1) Sample p% of the overall training data
            X_sample = X_train.sample(frac=(p / 100), axis=0)
            y_sample = y_train[X_sample.index]

            #   2) Fit linear model (including intercept) over sampled set
            lin_reg = LinearRegression(True)
            lin_reg.fit(X_sample, y_sample)

            #   3) Test fitted model over test set
            #   4) Store average and variance of loss over test set
            p_losses = np.append(p_losses, lin_reg.loss(X_test, y_test))

        mean = p_losses.mean()
        mean_s.append(mean)
        std = p_losses.std()

        lower_bound.append(mean - 2 * std)
        upper_bound.append((mean + 2 * std))

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    figure = go.Figure([
        go.Scatter(x=training_sizes, y=mean_s, line=dict(color='rgb(0,100,80)'), mode='lines'),
        go.Scatter(x=training_sizes + training_sizes[::-1],  # x, then x reversed
                   y=upper_bound + lower_bound[::-1],  # upper, then lower reversed
                   fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                   hoverinfo="skip",
                   showlegend=False
                   )
    ])
    figure.show()
