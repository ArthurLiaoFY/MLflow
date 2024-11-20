import numpy as np


def softmax(y: np.ndarray) -> np.ndarray:
    y_exp = np.array([np.exp(yi) for yi in y.squeeze()])
    return y_exp / y_exp.sum()
