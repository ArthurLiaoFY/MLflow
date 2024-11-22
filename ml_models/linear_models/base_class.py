from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import plotly
import plotly.graph_objects as go

from ml_models.linear_models.kernel_functions import softmax
from ml_models.linear_models.loss import root_mean_square_error
from ml_models.linear_models.metrics import r_square


class StatisticalModel:
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def plot(self):
        pass

    def plot_residual(self, X: np.ndarray, y: np.ndarray) -> None:
        residual = (self.predict(X=X) - y).squeeze()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=X.squeeze(),
                y=residual,
                mode="markers",
                name="Residuals",
            )
        )
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
        )
        fig.update_layout(
            title={
                "text": f"Residual Plot, RMSE: {root_mean_square_error(y_true=y, y_pred=self.predict(X=X)):.4f}",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            template="plotly_white",
            xaxis_title="X",
            yaxis_title="Residuals",
        )
        fig.show()

    def plot_fitted_value(self, X: np.ndarray, y: np.ndarray) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.predict(X=X),
                y=y,
                mode="markers",
            )
        )

        fig.update_layout(
            title={
                "text": f"Fitted Plot, R square: {r_square(y_true=y, y_pred=self.predict(X=X)):.4f}",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            template="plotly_white",
            xaxis_title="fitted value",
            yaxis_title="actual value",
        )
        fig.show()


class StatisticalTest:
    def __init__(self):
        pass


class LocalBaseModel:
    def __init__(
        self,
        kernel_func: Callable,
        bandwidth: float,
        num_of_knots: int,
        equal_space_knots: bool,
    ):
        self.kernel_func = kernel_func
        self.bandwidth = bandwidth
        self.num_of_knots = num_of_knots
        self.equal_space_knots = equal_space_knots
        self.beta_hat = {}
        self.fitted = False

    def get_knots(self, X: np.ndarray):
        return (
            np.linspace(start=X.min(), stop=X.max(), num=self.num_of_knots)
            if self.equal_space_knots
            else np.sort(np.unique(X.squeeze()))
        )

    def get_weight_matrix(self, X: np.ndarray, knot: float) -> np.ndarray:
        return np.diag(
            [self.kernel_func(u=x - knot, h=self.bandwidth) for x in X.squeeze()]
        )

    def _local_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        knot: float,
        to_model_matrix_func: Callable,
    ) -> None:
        model_matrix = to_model_matrix_func(X=X, knot=knot)
        weight_matrix = self.get_weight_matrix(X=X, knot=knot)
        self.beta_hat[knot] = (
            np.linalg.pinv(model_matrix.T @ weight_matrix @ model_matrix)
            @ model_matrix.T
            @ weight_matrix
            @ y
        )

    def _local_predict(
        self,
        xi: int,
        to_model_matrix_func: Callable,
    ) -> np.ndarray | None:
        knots_mask = np.logical_and(
            self.knots >= xi - self.bandwidth, self.knots <= xi + self.bandwidth
        )
        return (
            np.concatenate(
                [
                    to_model_matrix_func(X=np.array([xi]), knot=knot)
                    for knot in self.knots[knots_mask]
                ],
                axis=0,
            )
            * softmax(
                np.array(
                    [
                        self.kernel_func(u=xi - knot, h=self.bandwidth)
                        for knot in self.knots[knots_mask]
                    ],
                )
            )[:, np.newaxis]
            * np.concatenate(
                [self.beta_hat[knot][np.newaxis, :] for knot in self.knots[knots_mask]],
                axis=0,
            )
        ).sum()

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        if (X.ndim in (1, 2)) and X.shape[0] == y.shape[0]:
            self.knots = self.get_knots(X=X)
            for knot in self.knots:
                self._local_fit(
                    X=X,
                    y=y,
                    knot=knot,
                    to_model_matrix_func=to_model_matrix_func,
                )
            self.fitted = True
        else:
            return None

    def _predict(
        self,
        X: np.ndarray,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        return (
            np.array(
                [
                    self._local_predict(
                        xi=xi,
                        to_model_matrix_func=to_model_matrix_func,
                    )
                    for xi in X.squeeze()
                ]
            )
            if self.fitted
            else None
        )


class LinearBaseModel:
    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
        self.fitted = False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if (X.ndim in (1, 2)) and X.shape[0] == y.shape[0]:
            self.p = X.shape[-1]
            self.n = X.shape[0]
            self.deg_of_freedom = self.n - self.p
            if np.linalg.matrix_rank(X) < min(X.shape):
                print(
                    "The matrix is not invertible. Consider checking the input features."
                )
            inverse = np.linalg.pinv(X.T @ X)
            self.hat_matrix = X @ inverse @ X.T
            self.beta_hat = inverse @ X.T @ y
            self.fitted = True

            self.residuals = y - self._predict(X)
            self.regression_sum_of_squares = np.sum(self.residuals**2)
            self.sigma_hat = np.sqrt(
                self.regression_sum_of_squares / self.deg_of_freedom
            )
            self.cov_mat = inverse * self.sigma_hat**2

        else:
            return None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta_hat if self.fitted else None

    @property
    def leverages(self) -> np.ndarray | None:
        # focus on high leverages sample
        return np.diag(self.hat_matrix) if self.fitted else None

    @property
    def studentized_residuals(self) -> np.ndarray | None:
        # to check normality
        return (
            self.residuals.squeeze()
            / np.sqrt(np.diag(np.eye(self.hat_matrix.shape[0]) - self.hat_matrix))
            / self.sigma_hat
            if self.fitted
            else None
        )

    @property
    def cook_statistics(self) -> np.ndarray | None:
        # detecting influential sample
        return (
            (
                self.studentized_residuals.squeeze() ** 2
                * (
                    np.diag(self.hat_matrix)
                    / np.diag(np.eye(self.hat_matrix.shape[0]) - self.hat_matrix)
                )
                / self.p
            )
            if self.fitted
            else None
        )

    @property
    def jackknife_residuals(self) -> np.ndarray | None:
        # detect outliers sample
        return (
            self.studentized_residuals.squeeze()
            * np.sqrt(
                (self.n - self.p - 1)
                / (self.n - self.p - self.studentized_residuals.squeeze() ** 2)
            )
            if self.fitted
            else None
        )

    def plot_residual(self, index_name: str | None = None):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(self.n)),
                y=self.residuals.squeeze(),
                mode="markers",
                name="Residual",
                marker=dict(color="blue"),
                hovertemplate="Index: %{customdata}<br>Residual: %{y:.4f}<extra></extra>",
                customdata=(
                    index_name if index_name is not None else list(range(self.n))
                ),
            )
        )
        fig.add_hline(
            y=0.0,
            line_color="black",
        )
        fig.add_hline(
            y=self.sigma_hat,
            line_dash="dash",
            line_color="red",
        )
        fig.add_hline(
            y=-self.sigma_hat,
            line_dash="dash",
            line_color="red",
        )
        fig.update_layout(
            title="Residual Plot",
            xaxis_title="",
            yaxis_title="Residual",
            template="plotly_white",
        )

        fig.show()

    def plot_studentized_residual(self, index_name: str | None = None):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(self.n)),
                y=self.studentized_residuals.squeeze(),
                mode="markers",
                name="Studentized Residual",
                marker=dict(color="blue"),
                hovertemplate="Index: %{customdata}<br>Studentized Residual: %{y:.4f}<extra></extra>",
                customdata=(
                    index_name if index_name is not None else list(range(self.n))
                ),
            )
        )
        fig.add_hline(
            y=0.0,
            line_color="black",
        )
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="red",
        )
        fig.add_hline(
            y=-1,
            line_dash="dash",
            line_color="red",
        )
        fig.update_layout(
            title="Studentized Residual Plot",
            xaxis_title="",
            yaxis_title="Studentized Residual",
            template="plotly_white",
        )

        fig.show()

    def plot_leverage(self, index_name: str | None = None):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(self.n)),
                y=self.leverages.squeeze(),
                mode="markers",
                name="Leverage",
                marker=dict(color="blue"),
                hovertemplate="Index: %{customdata}<br>Leverage: %{y:.4f}<extra></extra>",
                customdata=(
                    index_name if index_name is not None else list(range(self.n))
                ),
            )
        )
        fig.add_hline(
            y=2 * self.p / self.n,
            line_dash="dash",
            line_color="red",
        )
        fig.update_layout(
            title="Leverage Plot",
            xaxis_title="",
            yaxis_title="Leverage",
            template="plotly_white",
        )

        fig.show()

    def plot_cook_statistics(self, index_name: str | None = None):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(self.n)),
                y=self.cook_statistics.squeeze(),
                mode="markers",
                name="Cook Statistic",
                marker=dict(color="blue"),
                hovertemplate="Index: %{customdata}<br>Cook Statistic: %{y:.4f}<extra></extra>",
                customdata=(
                    index_name if index_name is not None else list(range(self.n))
                ),
            )
        )

        fig.update_layout(
            title="Cook Statistic Plot",
            xaxis_title="",
            yaxis_title="Cook Statistic",
            template="plotly_white",
        )

        fig.show()

    def plot_jackknife_residual(self, index_name: str | None = None):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(self.n)),
                y=self.jackknife_residuals.squeeze(),
                mode="markers",
                name="Jackknife Residual",
                marker=dict(color="blue"),
                hovertemplate="Index: %{customdata}<br>Jackknife Residual: %{y:.4f}<extra></extra>",
                customdata=(
                    index_name if index_name is not None else list(range(self.n))
                ),
            )
        )

        fig.update_layout(
            title="Jackknife Residual Plot",
            xaxis_title="",
            yaxis_title="Jackknife Residual",
            template="plotly_white",
        )

        fig.show()

    def plot_normal_qq_plot(self):
        pass
