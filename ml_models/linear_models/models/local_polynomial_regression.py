from typing import Callable

import numpy as np

from ml_models.linear_models.tools import (
    to_local_constant_model_matrix,
    to_local_linear_model_matrix,
    to_local_polynomial_model_matrix,
)


def local_model_fit_check(fit: Callable):
    def wrapper(X: np.ndarray, y: np.ndarray):
        if (X.ndim == 1 or (X.ndim == 2 and X.shape[-1] == 1)) and X.shape[
            0
        ] == y.shape[0]:
            return fit(X=X, y=y if y.ndim == 2 else y[:, np.newaxis])
        else:
            return None

    return wrapper


class LocalModel:
    def __init__(self, kernel_func: Callable, num_of_knots: int = 51):
        self.kernel_func = kernel_func
        self.num_of_knots = num_of_knots
        self.beta_hat = {}

    def get_knots(self, X: np.ndarray):
        return np.linspace(start=X.min(), stop=X.max(), num=self.num_of_knots)

    def get_weight_matrix(self, X: np.ndarray, knot: float) -> np.ndarray:
        return np.diag([self.kernel_func(x - knot) for x in X.squeeze()])

    def local_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        knot: float,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        model_matrix = to_model_matrix_func(X=X, knot=knot)
        weight_matrix = self.get_weight_matrix(X=X)
        self.beta_hat[knot] = (
            np.linalg.pinv(model_matrix.T @ weight_matrix @ model_matrix)
            @ model_matrix.T
            @ weight_matrix
            @ y
        )

        return model_matrix @ self.beta_hat[knot]

    def local_predict(
        self,
        X: np.ndarray,
        knot: float,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        model_matrix = to_model_matrix_func(X=X, knot=knot)
        return model_matrix @ self.beta_hat[knot]

    @local_model_fit_check
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        self.knots = self.get_knots(X=X)
        for knot in self.knots:
            self.local_fit(
                X=X,
                y=y,
                knot=knot,
                to_model_matrix_func=to_model_matrix_func,
            )

    def predict(
        self,
        X: np.ndarray,
        to_model_matrix_func: Callable,
    ) -> np.ndarray:
        return np.array(
            [
                self.local_predict(
                    X=X,
                    knot=knot,
                    to_model_matrix_func=to_model_matrix_func,
                )
            ]
            for knot in self.knots
        ).sum(axis=1)


class LocalConstantModel(LocalModel):
    def __init__(
        self,
        kernel_func: Callable,
    ):
        super().__init__(kernel_func)

    def fit():
        pass

    def predict():
        pass


class LocalLinearModel(LocalModel):
    def __init__(
        self,
        kernel_func: Callable,
    ):
        super().__init__(kernel_func)

    def fit():
        pass

    def predict():
        pass


class LocalPolynomialModel(LocalModel):

    def __init__(
        self,
        kernel_func: Callable,
    ):
        super().__init__(kernel_func)

    def fit():
        pass

    def predict():
        pass
