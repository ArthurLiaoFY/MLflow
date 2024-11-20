# %%
import numpy as np


class PIDController:
    def __init__(
        self,
        target_point: float,
        learning_rate: float = 1,
        learning_rate_decay: float = 0.999,
        epsilon: float = 1e-7,
    ):
        self.reset_target_point(
            target_point=target_point,
            learning_rate=learning_rate,
        )
        self.learning_rate_decay = learning_rate_decay
        self.epsilon = epsilon

    def reset_target_point(self, target_point: float, learning_rate: float):
        self.target_point = target_point
        self.learning_rate = learning_rate

    def compute(self, y_new: float, Kp: np.ndarray):
        return Kp * (self.target_point - y_new)

    def estimate_Kp(self, X: np.ndarray, y: np.ndarray) -> None:
        beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.learning_rate *= self.learning_rate_decay
        return self.learning_rate / (beta_hat + self.epsilon)[1:, :]
