from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("/home/silver/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_perceptron(perc: Perceptron, sample: np.ndarray, response: np.ndarray ):
            losses.append(perc.loss(sample, y))

        perceptron = Perceptron(callback=loss_perceptron)
        perceptron.fit(X, y)


        # Plot figure of loss as function of fitting iteration
        x = np.arange(len(losses))
        figure = go.Figure(data=[go.Scatter(x=x, y=losses, name='Site #1')],
                           layout=go.Layout(xaxis_title="$\\text{Iteration}$", yaxis_title="$\\text{Loss}$",
                                            height=500))
        figure.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("/home/silver/IML.HUJI/datasets/" + f)

        # Fit models and predict over training set
        LDA_classifier = LDA()
        LDA_classifier.fit(X, y)
        LDA_pred = LDA_classifier.predict(X)

        bayes_classifier = GaussianNaiveBayes()
        bayes_classifier.fit(X, y)
        bayes_pred = bayes_classifier.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        symbols = np.array(["diamond", "circle", "square"])

        LDA_fig = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                             marker=dict(color=LDA_pred, symbol=symbols[y]))
        bayes_fig = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                               marker=dict(color=bayes_pred, symbol=symbols[y]))
        figure = make_subplots(rows=1, cols=2, subplot_titles=(f"Bayes accuracy {accuracy(y, bayes_pred)}",
                                                               f"LDA accuracy {accuracy(y, LDA_pred)}"))

        # Add traces for data-points setting symbols and colors
        figure.add_trace(LDA_fig, row=1, col=1)
        figure.add_trace(bayes_fig, row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        bayes_mu = bayes_classifier.mu_
        LDA_mu = LDA_classifier.mu_

        LDA_center = go.Scatter(x=bayes_mu[:, 0], y=bayes_mu[:, 1], mode="markers",
                             marker=dict(color="black", symbol="x"))
        bayes_center = go.Scatter(x=LDA_mu[:, 0], y=LDA_mu[:, 1], mode="markers",
                             marker=dict(color="black", symbol="x"))

        figure.add_trace(LDA_center, row=1, col=1)
        figure.add_trace(bayes_center, row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        LDA_ellipse = np.array([get_ellipse(bayes_mu[i], np.diag(bayes_classifier.vars_[i])) for i in
                                range(len(bayes_classifier.classes_))])
        bayes_elipse = np.array([get_ellipse(LDA_mu[i], LDA_classifier.cov_) for i in
                                range(len(LDA_classifier.classes_))])
        for e in LDA_ellipse:
            figure.add_trace(e, row=1, col=1)
        for e in bayes_elipse:
            figure.add_trace(e, row=1, col=2)
        figure.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
