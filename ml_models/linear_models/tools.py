import numpy as np


def to_model_matrix(X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    X = X if X.ndim == 2 else X[:, np.newaxis]
    return (
        np.concatenate(
            (
                np.ones(shape=(X.shape[0], 1)),
                X,
            ),
            axis=1,
        )
        if add_intercept
        else X
    )


def to_local_constant_model_matrix(X: np.ndarray, knot: float) -> np.ndarray:
    return np.ones(shape=(X.shape[0], 1))


def to_local_linear_model_matrix(X: np.ndarray, knot: float) -> np.ndarray:
    relative_distance = X - knot if X.ndim == 2 else X[:, np.newaxis] - knot
    return to_model_matrix(X=relative_distance)


def to_local_polynomial_model_matrix(X: np.ndarray, knot: float) -> np.ndarray:
    relative_distance = X - knot if X.ndim == 2 else X[:, np.newaxis] - knot
    return to_model_matrix(
        X=np.concatenate(
            (relative_distance, relative_distance**2),
            axis=1,
        )
    )
