# %%
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.stats import f, t


# %%
class LinearModel:
    def __init__(self, intercept: bool = True) -> None:
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
        self.n, self.p = x.shape
        self.deg_of_freedom = self.n - self.p - 1

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

        self.reg_sum_of_square = np.sum(self.residuals**2)

        self.sigma_hat = np.sqrt(self.reg_sum_of_square / self.deg_of_freedom)
        self.cov_mat = xtx_inv * np.pow(self.sigma_hat, 2)

        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        x = self.__add_intercept(x)
        return x @ self.beta_hat

    def summary(self) -> dict:
        if not self.fitted:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )
        self.null_reg_sum_of_square = np.pow(self.y - self.y.mean(), 2).sum()

        self.r_square = 1.0 - self.reg_sum_of_square / self.null_reg_sum_of_square

        self.adj_r_square = 1.0 - (
            (1 - self.r_square) * (self.n - 1) / self.deg_of_freedom
        )

        t_statistic = self.beta_hat / np.sqrt(np.diag(self.cov_mat))
        t_p_values = [
            2
            * min(
                1 - t.cdf(abs(t_stat), self.deg_of_freedom),
                t.cdf(abs(t_stat), self.deg_of_freedom),
            )
            for t_stat in t_statistic
        ]
        self.f_statistic = (
            (self.null_reg_sum_of_square - self.reg_sum_of_square)
            / self.p
            / self.reg_sum_of_square
            * (self.n - self.p - 1)
        )
        self.f_p_value = 1 - f.cdf(
            self.f_statistic, dfd=self.n - self.p - 1, dfn=self.p
        )

        row_idx = (
            ["intercept"] + self.feature_names if self.intercept else self.feature_names
        )
        return {
            "beta": {k: b for k, b in zip(row_idx, self.beta_hat)},
            "se(beta)": {k: b for k, b in zip(row_idx, np.sqrt(np.diag(self.cov_mat)))},
            "test_statistic": {k: b for k, b in zip(row_idx, t_statistic)},
            "p_value": {k: b for k, b in zip(row_idx, t_p_values)},
        }  # pd.DataFrame.from_dict(lm.summary(), orient="columns")

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

    def lm_report(self):
        pass
