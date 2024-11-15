# %%
import numpy as np


class PIDController:
    def __init__(
        self,
        target_point: float,
    ):
        self.epsilon = 1e-5
        self.target_point = target_point

    def compute(self, y_new: float, Kp: np.ndarray):
        return Kp * (self.target_point - y_new)

    def estimate_Kp(self, X: np.ndarray, y: np.ndarray) -> None:
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return np.maximum(
            1 / (beta_hat + self.epsilon),
            np.zeros_like(beta_hat),
        )[1:, :]
