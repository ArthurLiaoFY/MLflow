from typing import Callable

import numpy as np


def local_model_fit_check(fit_func: Callable):
    def wrapper(X: np.ndarray, y: np.ndarray):
        if (X.ndim == 1 or (X.ndim == 2 and X.shape[-1] == 1)) and X.shape[
            0
        ] == y.shape[0]:
            return fit_func(X=X, y=y if y.ndim == 2 else y[:, np.newaxis])
        else:
            return None

    return wrapper


def linear_model_fit_check(fit_func: Callable):
    def wrapper(X: np.ndarray, y: np.ndarray):
        if (X.ndim in (1, 2)) and X.shape[0] == y.shape[0]:
            return fit_func(
                X=X if X.ndim == 2 else X[:, np.newaxis],
                y=y if y.ndim == 2 else y[:, np.newaxis],
            )
        else:
            return None

    return wrapper
