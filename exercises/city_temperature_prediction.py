import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Loading the data
    X = pd.read_csv(filename, parse_dates=['Date'])

    # Preprocessing

    # Step 1: remove empty rows.
    X.dropna(inplace=True)

    # Step 2: clean up the data
    # Step 2.1: the data contains a whole bunch of temps set to -72.777778. Remove them.
    X = X[X['Temp'] > -72]

    # # Step 2.2.1: date breakdown:
    X['DayOfYear'] = X['Date'].dt.dayofyear

    # Step 2.2.2: remove date from existing data:
    X.drop(["Date"], inplace=True, axis=1)

    X.reset_index(inplace=True, drop=True)

    return X


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    filename = '/home/silver/IML.HUJI/datasets/City_Temperature.csv'
    X = load_data(filename)

    X['Year'] = X['Year'].astype(str)

    # Question 2 - Exploring data for specific country
    israel_data = X[X['Country'] == "Israel"]
    # figure = px.scatter(israel_data, x='DayOfYear', y='Temp', color='Year', title='Temperature in Israel as a Function of '
    #                                                                            'the Year')
    # figure.show()
    #
    # months = israel_data.groupby(israel_data['Month'])['Temp'].std()
    # figure = px.bar(months, title='STD of Temperature in Israel as a Function of the Month')
    # figure.show()

    # Question 3 - Exploring differences between countries
    grouped_std = X.groupby(['Country', 'Month'])['Temp'].std()
    grouped_average = X.groupby(['Country', 'Month'])['Temp'].mean()

    grouped_country = grouped_average.index.get_level_values(0)
    grouped_months = grouped_average.index.get_level_values(1)

    columns = {'Country': grouped_country, 'Month': grouped_months, 'std': grouped_std, 'Average': grouped_average}
    df = pd.DataFrame(columns)
    # figure = px.line(df, x='Month', y='Average', color='Country', error_y='std', title='Average Monthly Temperature')
    # figure.show()

    # Question 4 - Fitting model for different values of `k`
    # Step 1: Randomly split the dataset into a training set (75%) and test set (25%)
    y = israel_data.pop('Temp')

    israel_data.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)

    X_train, y_train, X_test, y_test = split_train_test(israel_data, y, 0.75)

    k_values = []
    losses = []

    for k in range(1, 11):
        # Step 2: For every value k ∈ [1,10], fit a polynomial model of degree k using the training set.
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(X_train["DayOfYear"], y_train)

        # Step 3: Record the loss of the model over the test set, rounded to 2 decimal places.
        loss = round(poly_fit.loss(X_test["DayOfYear"], y_test), 2)

        losses.append(loss)
        k_values.append(k)

        # Step 4: Print the test error recorded for each value of k
        print("for k=", k, " the loss is ", loss)

    # Step 5: plot the test error recorded for each value of k.
    # figure = px.bar(x=k_values, y=losses, title='Loss as a function of K')
    # figure.show()

    # Question 5 - Evaluating fitted model on different countries
    best_k = 5
    poly_fit = PolynomialFitting(best_k)
    poly_fit.fit(X_train["DayOfYear"], y_train)

    South_Africa = X[X['Country'] == "South Africa"]
    The_Netherlands = X[X['Country'] == "The Netherlands"]
    Jordan = X[X['Country'] == "Jordan"]

    South_Africa_loss = poly_fit.loss(South_Africa["DayOfYear"], South_Africa["Temp"])
    The_Netherlands_loss = poly_fit.loss(The_Netherlands["DayOfYear"], The_Netherlands["Temp"])
    Jordan_loss = poly_fit.loss(Jordan["DayOfYear"], Jordan["Temp"])

    losses = [South_Africa_loss, The_Netherlands_loss, Jordan_loss]
    data = {'Country': ['Jordan', 'South Africa', 'The Netherlands'], 'Loss': losses}
    countries = ['Jordan', 'South Africa', 'The Netherlands']

    figure = px.bar(x=countries, y=losses, title='Losses of Other Countries According to Israel Model')
    figure.show()

