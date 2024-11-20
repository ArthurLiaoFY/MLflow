from typing import Callable

import numpy as np

from ml_models.linear_models.base_class import LocalBaseModel, StatisticalModel
from ml_models.linear_models.tools import (
    to_local_constant_model_matrix,
    to_local_linear_model_matrix,
    to_local_polynomial_model_matrix,
)


class LocalConstantModel(LocalBaseModel, StatisticalModel):
    def __init__(
        self,
        bandwidth: int,
        kernel_func: Callable,
        num_of_knots: int = 51,
        equal_space_knots: bool = True,
    ):
        super().__init__(
            kernel_func=kernel_func,
            bandwidth=bandwidth,
            num_of_knots=num_of_knots,
            equal_space_knots=equal_space_knots,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._fit(
            X=X,
            y=y,
            to_model_matrix_func=to_local_constant_model_matrix,
        )

    def predict(self, X: np.ndarray) -> np.ndarray | None:
        return self._predict(
            X=X,
            to_model_matrix_func=to_local_constant_model_matrix,
        )


class LocalLinearModel(LocalBaseModel, StatisticalModel):
    def __init__(
        self,
        bandwidth: int,
        kernel_func: Callable,
        num_of_knots: int = 51,
        equal_space_knots: bool = True,
    ):
        super().__init__(
            kernel_func=kernel_func,
            bandwidth=bandwidth,
            num_of_knots=num_of_knots,
            equal_space_knots=equal_space_knots,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._fit(
            X=X,
            y=y,
            to_model_matrix_func=to_local_linear_model_matrix,
        )

    def predict(self, X: np.ndarray) -> np.ndarray | None:
        return self._predict(
            X=X,
            to_model_matrix_func=to_local_linear_model_matrix,
        )


class LocalPolynomialModel(LocalBaseModel, StatisticalModel):

    def __init__(
        self,
        bandwidth: int,
        kernel_func: Callable,
        num_of_knots: int = 51,
        equal_space_knots: bool = True,
    ):
        super().__init__(
            kernel_func=kernel_func,
            bandwidth=bandwidth,
            num_of_knots=num_of_knots,
            equal_space_knots=equal_space_knots,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._fit(
            X=X,
            y=y,
            to_model_matrix_func=to_local_polynomial_model_matrix,
        )

    def predict(self, X: np.ndarray) -> np.ndarray | None:
        return self._predict(
            X=X,
            to_model_matrix_func=to_local_polynomial_model_matrix,
        )
