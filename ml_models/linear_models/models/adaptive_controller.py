# %%
import numpy as np

from .linear_model import LinearModel


class BetaController:
    def __init__(
        self,
        target_point: float,
        epsilon: float = 1e-7,
    ):
        self.lm = LinearModel(add_intercept=True)
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

    def estimate_parameter(self, X: np.ndarray, y: np.ndarray) -> None:
        self.lm.fit(X=X, y=y)

        self.intercept = self.lm.beta_hat[0].item()
        self.trend = self.lm.beta_hat[1:].squeeze()

        self.Kp = 1 / (self.trend + self.epsilon)


class PIDController:
    def __init__(
        self,
        target_point: float,
        Kp: float,
        Ki: float,
        Kd: float,
        tau: float = 0.1,
        dt: float = 1.0,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.tau = tau
        self.dt = dt
        self.reset_target_point(target_point=target_point)

    def reset_target_point(self, target_point: float):
        self.target_point = target_point
        self.prev_error = 0
        self.integral = 0
        self.prev_d = 0

    def compute(self, y_new: float):
        error = self.target_point - y_new
        self.integral += error * self.dt

        derivative = (
            (error - self.prev_error) / self.dt
            if self.tau is None
            else (self.tau * self.prev_d + (error - self.prev_error))
            / (self.tau + self.dt)
        )

        ctrl = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error
        self.prev_d = derivative

        return ctrl

    def estimate_parameter(self):
        # use Ziegler-Nichols Method
        Kp = 0
        Ki = 0
        Kd = 0

        pass
