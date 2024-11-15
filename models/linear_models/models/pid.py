# %%
import numpy as np


class PIDController:
    def __init__(self, target_point: float):
        self.Kp = None
        self.epsilon = 1e-5
        self.target_point = target_point

    def compute(self, y_new: float):
        return self.Kp * (self.target_point - y_new)

    def update_Kp(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.Kp is None:
            self.Kp = np.ones(shape=(X.shape[-1] - 1))
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

        self.Kp = np.max(
            np.concatenate(
                (
                    1 / (beta_hat + self.epsilon),
                    np.zeros_like(beta_hat),
                ),
                axis=0,
            ),
            axis=0,
        )

