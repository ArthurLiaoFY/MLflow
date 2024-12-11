# %%
import numpy as np


class MahalanobisDistance:
    def __init__(self) -> None:
        self.fitted = False

    def inner_distance(self, X: np.ndarray) -> np.ndarray:
        self.sample_mean = X.mean(axis=0)
        self.inverse_sample_covariance_matrix = np.linalg.pinv(
            (X - self.sample_mean).T @ (X - self.sample_mean) / (len(X) - 1)
        )
        self.fitted = True
        return np.diag(
            (X - self.sample_mean)
            @ self.inverse_sample_covariance_matrix
            @ (X - self.sample_mean).T
        )

    def outer_distance(self, xi: np.ndarray) -> np.ndarray | None:
        return (
            np.diag(
                (xi - self.sample_mean)
                @ self.inverse_sample_covariance_matrix
                @ (xi - self.sample_mean).T
            )
            if self.fitted
            else None
        )
