from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import plotly.io as pio

pio.templates.default = "simple_white"
from utils import *


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    num_of_samples = 1000

    sample = np.random.normal(mu, sigma, num_of_samples)

    estimator = UnivariateGaussian(False)
    estimator.fit(sample)

    print("(", estimator.mu_, ",", estimator.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    means = np.array([])  # initialize array where we'll be saving the abs distance

    abs_dist_estimator = UnivariateGaussian(False)

    jump = 10
    for i in range(jump, 1010, jump):
        new_sample = sample[:i]  # take sample of desired size
        abs_dist_estimator.fit(new_sample)
        means = np.append(means, np.abs(abs_dist_estimator.mu_ - mu))

    indexes = np.arange(jump, 1010, jump)
    go.Figure([go.Scatter(x=indexes, y=means, mode='markers+lines', name=r'$\widehat\sigma^2$')],
              layout=go.Layout(title=r"$\text{ Absolute Distance Between the Estimated and True Value of the "
                                     r"Expectation As Function Of Sample Size}$", xaxis_title=r"$\text{ Sample Size}$",
                               yaxis_title=r"$\text{ Absolute Distance}$", height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = np.array([])  # initialize array where we'll be saving the pdfs

    for s in sample:  # iterate over samples
        pdfs = np.append(pdfs, estimator.pdf(s))

    go.Figure([go.Scatter(x=sample, y=pdfs, mode='markers', name=r'$\widehat\sigma^2$')],
              layout=go.Layout(title=r"$\text{ Empirical PDF of Model Fitted in Question 1}$",
                               axis_title=r"$\text{ Sample Value}$", yaxis_title=r"$\text{ PDF of Sample}$",
                               height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    num_of_samples = 1000
    sample = np.random.multivariate_normal(mu, sigma, num_of_samples)

    estimator = MultivariateGaussian()
    estimator.fit(sample)

    print("Estimated Expectation: \n", estimator.mu_)
    print("Estimated Covariance Matrix: \n", estimator.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    f2 = np.empty((200, 200))

    # for Q6:
    max_likelihood = 0
    max_f1_val = 0
    max_f3_val = 0

    for i, n1 in enumerate(f1):
        for j, n2 in enumerate(f3):
            new_mu = np.array([n1, 0, n2, 0])
            k = estimator.log_likelihood(new_mu, sigma, sample)
            f2[i][j] = k

            # for Q6
            if max_likelihood == 0:
                max_likelihood = k
                max_f1_val = n1
                max_f3_val = n2
            elif k > max_likelihood:
                max_likelihood = k
                max_f1_val = n1
                max_f3_val = n2

    fig = go.Figure(data=go.Heatmap(z=f2, x=f1, y=f3, hoverongaps=False))
    fig.show()

    # Question 6 - Maximum likelihood
    print("(", round(max_f1_val, 3), ",", round(max_f3_val, 3), ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

