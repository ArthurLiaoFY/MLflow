# %%
import numpy as np
import pandas as pd
from scipy.stats import t

import plotly.graph_objects as go


# %%
class LinearModel:
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.feature_names = None
        self.x = None
        self.y = None
        self.beta = None
        self.residuals = None
        self.cov_mat = None
        self.regression_dum_of_square = None

    @staticmethod
    def _prepare_input(data: np.ndarray | pd.DataFrame) -> np.array:
        return np.squeeze(data) if isinstance(data, np.ndarray) else np.squeeze(data.values)

    def _add_intercept(self, x: np.ndarray) -> np.array:
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1) if self.intercept else x

    def fit(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        self.feature_names = None if type(x) is np.array else list(x.columns)

        self.x, self.y = self._add_intercept(self._prepare_input(x)), self._prepare_input(y)
        if np.linalg.matrix_rank(self.x) < min(self.x.shape):
            print('matrix is not invertible')

        hat_matrix = np.linalg.inv(self.x.T @ self.x) @ self.x.T
        self.beta = hat_matrix @ self.y
        self.residuals = (self.y - self.x @ self.beta)[:, np.newaxis]
        self.regression_dum_of_square = (self.residuals ** 2).sum()
        self.cov_mat = np.sqrt(
            np.linalg.inv(self.x.T @ self.x) * (self.residuals.T @ self.residuals / (self.x.shape[0] - self.x.shape[1]))
        )

    def plot_residual(self):
        # normal QQ plot (new py script)
        # leverage point
        # outlier
        # influential point
        # residual plot
        try:
            fig = go.Figure(go.Scatter(x=list(range(len(self.residuals))), y=self.residuals, mode='markers'))
            fig.update_layout(title_text='Residual Plot', title_x=0.5)
            fig.show()
        except ValueError:
            print('model fit must be called before plot residual')

    def plot_xy_plot(self):
        # return
        pass

    def summary(self) -> pd.DataFrame:
        summary_table_idx = ['intercept'] + self.feature_names if self.intercept else self.feature_names
        return pd.DataFrame(
            index=summary_table_idx,
            columns=['beta', 'se(beta)', 'test_statistic', 'p_value'],
            data=np.array(
                [
                    self.beta,
                    np.diag(self.cov_mat),
                    self.beta / np.diag(self.cov_mat),
                    [
                        2
                        * min(
                            1 - t.cdf(t_stat, self.x.shape[0] - self.x.shape[1]),
                            t.cdf(t_stat, self.x.shape[0] - self.x.shape[1]),
                        )
                        for t_stat in self.beta / np.diag(self.cov_mat)
                    ],
                ]
            ).T,
        )

    def predict(self, x: np.ndarray | pd.DataFrame):
        try:
            pred = x @ self.beta if hasattr(self.beta, 'beta') else None
        except ValueError:
            pred = None

        return pred


def sum_coding_matrix(
        data: np.ndarray | pd.Series, prefix: str = '', prefix_sep: str = '_', baseline: int | str | None = None
):
    data = data.values if type(data) is pd.Series else data
    unique_cats = sorted(np.unique(data).astype('str'))
    if str(baseline) not in unique_cats or baseline is None:
        baseline = unique_cats[-1]
        print('set baseline : ', baseline)
    else:
        baseline = str(baseline)

    sum_coding_df_columns = [prefix + prefix_sep + unique_cat for unique_cat in unique_cats if unique_cat != baseline]
    sum_coding = pd.DataFrame(columns=sum_coding_df_columns, data=np.zeros((len(data), len(unique_cats) - 1)))

    for idx, cat in enumerate(data):
        if str(cat) == baseline:
            sum_coding.loc[idx, sum_coding_df_columns] = -1

        else:
            sum_coding.loc[idx, prefix + prefix_sep + str(cat)] = 1

    return sum_coding


# %%

ex_data = pd.DataFrame(
    columns=['A', 'B', 'C', 'y'],
    data=[
        ['1', '1', '1', 19.25],
        ['1', '1', '1', 19.89],
        ['1', '1', '1', 20.1],
        ['1', '2', '2', 19.01],
        ['1', '2', '2', 20.3],
        ['1', '2', '2', 20],
        ['1', '3', '3', 19.36],
        ['1', '3', '3', 19.96],
        ['1', '3', '3', 18.24],
        ['2', '1', '2', 19.79],
        ['2', '1', '2', 19.01],
        ['2', '1', '2', 20.85],
        ['2', '2', '3', 18.75],
        ['2', '2', '3', 20.93],
        ['2', '2', '3', 20.36],
        ['2', '3', '1', 19.98],
        ['2', '3', '1', 19.54],
        ['2', '3', '1', 19.44],
        ['3', '1', '3', 19.21],
        ['3', '1', '3', 21.3],
        ['3', '1', '3', 19.45],
        ['3', '2', '1', 19.01],
        ['3', '2', '1', 20.3],
        ['3', '2', '1', 20],
        ['3', '3', '2', 19.36],
        ['3', '3', '2', 19.96],
        ['3', '3', '2', 18.24],
    ],
)
ex_data = (
    ex_data.groupby(['A', 'B', 'C'])['y']
    .apply(lambda x: 10 * np.log10(x.mean() ** 2 / x.std() ** 2))
    .reset_index()
    .drop_duplicates()
)

# %%
model_X = pd.concat(
    objs=(
        sum_coding_matrix(data=ex_data['A'], prefix='A', prefix_sep='_'),
        sum_coding_matrix(data=ex_data['B'], prefix='B', prefix_sep='_'),
        sum_coding_matrix(data=ex_data['C'], prefix='C', prefix_sep='_'),
    ),
    axis=1,
).astype(float)
# %%
lm = LinearModel()
lm.fit(x=model_X, y=ex_data['y'])
lm.plot_residual()
# %%
lm.summary()
