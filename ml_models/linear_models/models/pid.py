# %%
import numpy as np


class PIDController:
    def __init__(
        self,
        target_point: float,
        epsilon: float = 1e-7,
    ):
        self.reset_target_point(
            target_point=target_point,
        )
        self.epsilon = epsilon

    def reset_target_point(self, target_point: float):
        self.target_point = target_point

    def compute(self, y_new: float, Kp: np.ndarray):
        return Kp * (self.target_point - y_new)

    def estimate_Kp(self, X: np.ndarray, y: np.ndarray) -> None:
        beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
        return 1 / (beta_hat + self.epsilon)[1:, :]
