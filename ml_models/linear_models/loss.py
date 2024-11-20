import numpy as np


def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def weighted_log_likelihood_loss(
    y_true: np.ndarray, y_pred: np.ndarray, weight: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    prob = 1.0 / (1.0 + np.exp(-y_pred))

    grad = (1 - weight) * (1 - y_true) * prob - weight * y_true * (1 - prob)

    hess = prob * (1.0 - prob)
    hess *= weight * y_true + (1 - weight) * (1 - y_true)

    return grad, hess
