# %%
import numpy as np


def mahalanobis_distance(X: np.ndarray):
    return np.diag(
        (X - X.mean(axis=0))
        @ np.linalg.pinv((X - X.mean(axis=0)).T @ (X - X.mean(axis=0)) / (len(X) - 1))
        @ (X - X.mean(axis=0)).T
    )
