# %%
import numpy as np


class PIDController:
    def __init__(
        self,
        target_point: float,
        learning_rate: float = 0.05,
        learning_rate_decay: float = 0.99,
        epsilon: float = 1e-7,
    ):
        self.target_point = target_point
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epsilon = epsilon

        self.prev_beta_hat = None

    def compute(self, y_new: float, Kp: np.ndarray):
        return Kp * (self.target_point - y_new)

    def estimate_Kp(self, X: np.ndarray, y: np.ndarray) -> None:
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        if self.prev_beta_hat is None:
            self.prev_beta_hat = beta_hat
            pass

        else:
            self.prev_beta_hat += self.learning_rate * (beta_hat - self.prev_beta_hat)
            self.learning_rate *= self.learning_rate_decay

        return np.maximum(
            1 / (self.prev_beta_hat + self.epsilon),
            np.zeros_like(self.prev_beta_hat),
        )[1:, :]
