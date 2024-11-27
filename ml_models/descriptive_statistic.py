import numpy as np


def sample_mean(X: np.ndarray):
    return X.mean(axis=0)


def sample_variance(X: np.ndarray):
    return (X - sample_mean(X).T) @ (X - sample_mean(X).T) / (len(X) - 1)


def sample_covariance(X1: np.ndarray, X2: np.ndarray):
    return (X1 - sample_mean(X1).T) @ (X2 - sample_mean(X2).T) / (len(X1) - 1)
