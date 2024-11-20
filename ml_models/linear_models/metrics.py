import numpy as np


def r_square(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - (
        np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    )
