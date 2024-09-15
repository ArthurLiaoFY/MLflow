# %%
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import t


# %%
class LinearModel:
    def __init__(self, intercept: bool = True):
        self.intercept = intercept
        self.fitted = False

    def __add_intercept(self, x: np.ndarray) -> np.ndarray:
        if self.intercept:
            return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return x

    def fit(
        self, x: np.ndarray, y: np.ndarray, f_names: Optional[List[str]] = None
    ) -> None:
        self.feature_names = (
            f_names if f_names is not None else [f"f{i}" for i in range(x.shape[1])]
        )
        self.model_matrix = self.__add_intercept(x)
        self.y = y

        if np.linalg.matrix_rank(self.model_matrix) < min(self.model_matrix.shape):
            raise ValueError(
                "The matrix is not invertible. Consider checking the input features."
            )

        # Use pseudo-inverse for numerical stability
        xtx_inv = np.linalg.pinv(self.model_matrix.T @ self.model_matrix)
        self.beta_hat = xtx_inv @ self.model_matrix.T @ self.y
        self.residuals = self.y - self.model_matrix @ self.beta_hat
        self.regression_dum_of_square = np.sum(self.residuals**2)
        self.deg_of_freedom = (
            self.model_matrix.shape[0] - self.model_matrix.shape[1]
        )  # n - p
        self.sigma_hat = np.sqrt(self.regression_dum_of_square / self.deg_of_freedom)

        # Covariance matrix
        self.cov_mat = xtx_inv * np.pow(self.sigma_hat, 2)
        self.fitted = True

    def plot_residual(self) -> None:
        if not self.fitted:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        fig = go.Figure(
            go.Scatter(
                x=np.arange(len(self.residuals)), y=self.residuals, mode="markers"
            )
        )
        fig.update_layout(
            title_text="Residual Plot",
            title_x=0.5,
            xaxis_title="Index",
            yaxis_title="Residuals",
        )
        fig.show()

    def summary(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        summary_table_idx = (
            ["intercept"] + self.feature_names if self.intercept else self.feature_names
        )
        test_statistic = self.beta_hat / np.sqrt(np.diag(self.cov_mat))
        p_values = [
            2
            * min(
                1 - t.cdf(abs(t_stat), self.deg_of_freedom),
                t.cdf(abs(t_stat), self.deg_of_freedom),
            )
            for t_stat in test_statistic
        ]

        return pd.DataFrame(
            {
                "beta": self.beta_hat,
                "se(beta)": np.sqrt(np.diag(self.cov_mat)),
                "test_statistic": test_statistic,
                "p_value": p_values,
            },
            index=summary_table_idx,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        x = self.__add_intercept(x)
        return x @ self.beta_hat


# %%
iris = pd.read_csv("iris.csv")
# %%
lm = LinearModel()
lm.fit(
    x=iris[["Sepal.Width", "Petal.Length", "Petal.Width"]],
    y=iris["Sepal.Length"].to_numpy(),
    f_names=["Sepal.Width", "Petal.Length", "Petal.Width"],
)
lm.plot_residual()
# %%
lm.summary()

# %%
