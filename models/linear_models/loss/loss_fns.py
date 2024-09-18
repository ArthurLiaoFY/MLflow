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


def weighted_log_likelihood_loss(
    y_true, y_pred, weight: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    # Precompute constants
    prob = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid function

    # Compute gradients
    grad = (1 - weight) * (1 - y_true) * prob - weight * y_true * (1 - prob)

    # Compute Hessians
    hess = prob * (1.0 - prob)  # common term
    hess *= weight * y_true + (1 - weight) * (1 - y_true)  # element-wise multiply

    return grad, hess
