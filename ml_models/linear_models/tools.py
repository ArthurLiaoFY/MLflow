import numpy as np


def to_model_matrix(x: np.ndarray) -> np.ndarray:
    x = x if x.ndim == 2 else x[:, np.newaxis]
    return np.concatenate(
        (
            np.ones(shape=(x.shape[0], 1)),
            x,
        ),
        axis=1,
    )
