from typing import Callable

import numpy as np

from ml_models.linear_models.base_class import (
    LinearBaseModel,
    StatisticalModel,
    StatisticalTest,
)
from ml_models.linear_models.tools import to_model_matrix


class LinearModel(LinearBaseModel, StatisticalModel):
    def __init__(self, add_intercept: bool = True):
        super().__init__(add_intercept)

    def fit(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarray | None = None
    ) -> np.ndarray:
        return self._fit(
            X=X,
            y=y,
            W=W,
        )

    def predict(self, X: np.ndarray) -> np.ndarray | None:
        return self._predict(
            X=X,
        )


class ANOVA(LinearBaseModel, StatisticalTest):
    def __init__(self):
        super().__init__(add_intercept=True)
