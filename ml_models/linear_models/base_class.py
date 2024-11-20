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

    def _fit(self, X: np.ndarray, y: np.ndarray, W: np.ndarray | None = None) -> None:
        if (X.ndim in (1, 2)) and X.shape[0] == y.shape[0]:
            self.W = (
                W if W.ndim == 2 else np.eye(W) if W.ndim == 1 else np.eye(X.shape[-1])
            )
            self.beta_hat = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
            self.fitted = True
        else:
            return None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta_hat
