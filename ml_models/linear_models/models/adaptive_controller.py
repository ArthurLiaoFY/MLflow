# %%
import numpy as np


class BetaController:
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

    def compute(self, y_new: float):
        return (
            self.Kp
            * (self.target_point - y_new)
            * np.abs(self.Kp).squeeze()
            / np.abs(self.Kp).sum()
        )

    def estimate_Kp(self, X: np.ndarray, y: np.ndarray) -> None:
        beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.Kp = 1 / (beta_hat + self.epsilon)[1:, :]


class PIDController:
    def __init__(
        self,
        target_point: float,
        Kp: float,
        Ki: float,
        Kd: float,
        tau: float | None = None,
        dt: float = 1.0,
    ):
        self.target_point = target_point
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.tau = tau
        self.dt = dt

        self.prev_error = 0
        self.integral = 0
        self.prev_d = 0

    def reset_target_point(self, target_point: float):
        self.target_point = target_point

    def compute(self, y_new: float):
        error = self.target_point - y_new
        self.integral += error * self.dt

        derivative = (
            (error - self.prev_error) / self.dt
            if self.tau is None
            else (self.tau * self.prev_d + self.dt * derivative) / (self.tau + self.dt)
        )

        ctrl = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error
        self.prev_d = derivative

        return ctrl
