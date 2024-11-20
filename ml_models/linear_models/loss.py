import numpy as np
from sklearn.metrics import confusion_matrix

# print(
#     'Confusion Matrix : \n\n',
#     pd.DataFrame(
#         index=pd.MultiIndex.from_tuples((('Actual', 'Fail'), ('Actual', 'Pass'))),
#         columns=pd.MultiIndex.from_tuples((('Predict', 'Fail'), ('Predict', 'Pass'))),
#         data=confusion_matrix(y_true=valid_y, y_pred=pred_y),
#     ),
# )


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
