from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import plotly
import plotly.graph_objects as go

from .decorator import linear_model_fit_check, local_model_fit_check


class StatisticalModel:
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def plot_residual(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        residual = (self.predict(X=X) - y).squeeze()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=X.squeeze(),
                y=(self.predict(X=X) - y).squeeze(),
            )
        )
        fig.show()


class LocalBaseModel:
    def __init__(self, kernel_func: Callable, num_of_knots: int = 51):
        self.kernel_func = kernel_func
        self.num_of_knots = num_of_knots
        self.beta_hat = {}
        self.fitted = False

    def get_knots(self, X: np.ndarray):
        return np.linspace(start=X.min(), stop=X.max(), num=self.num_of_knots)

    def get_weight_matrix(self, X: np.ndarray, knot: float) -> np.ndarray:
        return np.diag([self.kernel_func(x - knot) for x in X.squeeze()])

    def __local_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        knot: float,
        to_model_matrix_func: Callable,
    ) -> None:
        model_matrix = to_model_matrix_func(X=X, knot=knot)
        weight_matrix = self.get_weight_matrix(X=X)
        self.beta_hat[knot] = (
            np.linalg.pinv(model_matrix.T @ weight_matrix @ model_matrix)
            @ model_matrix.T
            @ weight_matrix
            @ y
        )

    def __local_predict(
        self,
        X: np.ndarray,
        knot: float,
        to_model_matrix_func: Callable,
    ) -> np.ndarray | None:
        model_matrix = to_model_matrix_func(X=X, knot=knot)
        return model_matrix @ self.beta_hat[knot]

    @local_model_fit_check
    def __fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        self.knots = self.get_knots(X=X)
        for knot in self.knots:
            self.__local_fit(
                X=X,
                y=y,
                knot=knot,
                to_model_matrix_func=to_model_matrix_func,
            )
        self.fitted = True

    def __predict(
        self,
        X: np.ndarray,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        return (
            np.array(
                [
                    self.__local_predict(
                        X=X,
                        knot=knot,
                        to_model_matrix_func=to_model_matrix_func,
                    )
                ]
                for knot in self.knots
            ).sum(axis=1)
            if self.fitted
            else None
        )


class LinearBaseModel:
    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
        self.fitted = False

    @linear_model_fit_check
    def __fit(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarrayNone = None
    ) -> np.ndarray: | 
        W = np.eye(X.shape[-1]) if W is None else W
        self.beta_hat = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        self.fitted = True

    def __predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta_hat
