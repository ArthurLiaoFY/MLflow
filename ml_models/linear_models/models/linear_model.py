# from typing import Callable

# from scipy.stats import f, t
import numpy as np

from ml_models.descriptive_statistic import sample_covariance, sample_variance
from ml_models.linear_models.base_class import (
    LinearBaseModel,
    StatisticalModel,
    StatisticalTest,
)
from ml_models.linear_models.tools import to_model_matrix


class LinearModel(LinearBaseModel, StatisticalModel):
    def __init__(self, add_intercept: bool = True):
        super().__init__(add_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._fit(
            X=to_model_matrix(X=X, add_intercept=self.add_intercept),
            y=y,
        )

    def predict(self, X: np.ndarray) -> np.ndarray | None:
        return self._predict(
            X=to_model_matrix(X=X, add_intercept=self.add_intercept),
        )


class CanonicalCorrelation:
    def __init__(self):
        pass

    def fit(self, X1: np.ndarray, X2: np.ndarray):
        S_11_inv = np.linalg.pinv(sample_variance(X1))
        S_12 = sample_covariance(X1, X2)
        S_22_inv = np.linalg.pinv(sample_variance(X2))

        S_11_eigen_value, S_11_eigen_vector = np.linalg.eig(S_11_inv)

        return np.linalg.eig(
            a=S_11_eigen_vector
            @ np.diag(np.sqrt(S_11_eigen_value))
            @ S_11_eigen_vector.T
            @ S_12
            @ S_22_inv
            @ S_12.T
            @ S_11_eigen_vector
            @ np.diag(np.sqrt(S_11_eigen_value))
            @ S_11_eigen_vector.T
        )


class ANOVA(LinearBaseModel, StatisticalTest):
    def __init__(self):
        super().__init__(add_intercept=True)


# %%

# %%


# class old_LinearModel:
#     def __init__(self, intercept: bool = True) -> None:
#         self.intercept = intercept
#         self.fitted = False

#     def __add_intercept(
#         self,
#         x: np.ndarray,
#     ) -> np.ndarray:
#         if self.intercept:
#             return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
#         return x

#     def fit(
#         self,
#         x: np.ndarray,
#         y: np.ndarray,
#         f_names: Optional[List[str]] = None,
#     ) -> None:
#         self.feature_names = (
#             f_names if f_names is not None else [f"f{i}" for i in range(x.shape[1])]
#         )
#         self.n, self.p = x.shape
#         self.deg_of_freedom = self.n - self.p - 1

#         self.model_matrix = self.__add_intercept(x)
#         self.y = y

#         if np.linalg.matrix_rank(self.model_matrix) < min(self.model_matrix.shape):
#             raise ValueError(
#                 "The matrix is not invertible. Consider checking the input features."
#             )

#         # Use pseudo-inverse for numerical stability
#         xtx_inv = np.linalg.pinv(self.model_matrix.T @ self.model_matrix)
#         self.beta_hat = xtx_inv @ self.model_matrix.T @ self.y
#         self.residuals = self.y - self.model_matrix @ self.beta_hat

#         self.reg_sum_of_square = np.sum(self.residuals**2)

#         self.sigma_hat = np.sqrt(self.reg_sum_of_square / self.deg_of_freedom)
#         self.cov_mat = xtx_inv * np.pow(self.sigma_hat, 2)

#         self.fitted = True

#     def predict(
#         self,
#         x: np.ndarray,
#     ) -> np.ndarray:
#         if not self.fitted:
#             raise ValueError(
#                 "The model has not been fitted yet. Call the fit method first."
#             )

#         x = self.__add_intercept(x)
#         return x @ self.beta_hat

#     def summary(self) -> dict:
#         if not self.fitted:
#             raise ValueError(
#                 "The model has not been fitted yet. Call the fit method first."
#             )
#         self.null_reg_sum_of_square = np.pow(self.y - self.y.mean(), 2).sum()

#         self.r_square = 1.0 - self.reg_sum_of_square / self.null_reg_sum_of_square

#         self.adj_r_square = 1.0 - (
#             (1 - self.r_square) * (self.n - 1) / self.deg_of_freedom
#         )

#         t_statistic = self.beta_hat / np.sqrt(np.diag(self.cov_mat))
#         t_p_values = [
#             2
#             * min(
#                 1 - t.cdf(abs(t_stat), self.deg_of_freedom),
#                 t.cdf(abs(t_stat), self.deg_of_freedom),
#             )
#             for t_stat in t_statistic
#         ]
#         self.f_statistic = (
#             (self.null_reg_sum_of_square - self.reg_sum_of_square)
#             / self.p
#             / self.reg_sum_of_square
#             * (self.n - self.p - 1)
#         )
#         self.f_p_value = 1 - f.cdf(
#             self.f_statistic, dfd=self.n - self.p - 1, dfn=self.p
#         )

#         row_idx = (
#             ["intercept"] + self.feature_names if self.intercept else self.feature_names
#         )
#         return {
#             "beta": {k: b for k, b in zip(row_idx, self.beta_hat)},
#             "se(beta)": {k: b for k, b in zip(row_idx, np.sqrt(np.diag(self.cov_mat)))},
#             "test_statistic": {k: b for k, b in zip(row_idx, t_statistic)},
#             "p_value": {k: b for k, b in zip(row_idx, t_p_values)},
#         }  # pd.DataFrame.from_dict(lm.summary(), orient="columns")

#     def plot_residual(self) -> None:
#         if not self.fitted:
#             raise ValueError(
#                 "The model has not been fitted yet. Call the fit method first."
#             )

#         fig = go.Figure(
#             go.Scatter(
#                 x=np.arange(len(self.residuals)), y=self.residuals, mode="markers"
#             )
#         )
#         fig.update_layout(
#             title_text="Residual Plot",
#             title_x=0.5,
#             xaxis_title="Index",
#             yaxis_title="Residuals",
#         )
#         fig.show()

#     def lm_report(self):
#         pass


# class Nested_ANOVA:
#     """
#     This object using Nested ANOVA analysis to find significant factors, and it'll return Nested ANOVA summary table.
#     """

#     def __init__(self, Y, X, DataFrame):
#         """
#         Parameters
#         -----
#         Y : dict
#             1st input argument. Protected
#         X : dict
#             2st input argument. Protected
#         DataFrame : DataFrame
#             3rd input argument. Public
#         """
#         self.Y = Y
#         assert type(self.Y) == dict, "Y must be a dict"
#         self.X = X
#         assert type(self.X) == dict, "X must be a dict"
#         self.DataFrame = DataFrame.copy()
#         assert type(self.DataFrame) == pd.core.frame.DataFrame
#         assert (
#             tuple(self.Y.items())[0] in self.DataFrame.columns
#         ), "the dataframe you import must contain {}".format(self.Y)
#         self.DataFrame.columns = self.DataFrame.columns.map("_".join)
#         assert (
#             self.DataFrame["_".join(list(self.Y.items())[0])].dtype != object
#         ), "The data type of {} must be 'float' or 'int'".format(self.Y)
#         self.FormulaDict = dict()
#         self.__ColumnList = [
#             "Sum of Squares",
#             "df",
#             "Mean Square",
#             "F Stat",
#             "P-value",
#             "    ",
#         ]
#         self.ANOVA_table = pd.DataFrame(columns=self.__ColumnList)
#         self.__Prefix = ""
#         self.__InteractionPrefix = ""
#         self.__PreviousFormula = "_".join(list(self.Y.items())[0]) + " ~ 1"
#         self.__PreviousFittedValues = LinearModel(
#             self.__PreviousFormula, self.DataFrame
#         ).FittedValues
#         self.__PreviousDf = 0
#         self.fit(self.X)

#     def fit(self, Dict):
#         for Key, Values in Dict.items():
#             if self.__Prefix != "":
#                 self.ANOVA_table = self.ANOVA_table.append(
#                     pd.DataFrame(index=[Key], columns=self.__ColumnList)
#                 )
#                 if isinstance(Values, dict):
#                     self.__PreviousFormula += (
#                         " + " + self.__InteractionPrefix + self.__Prefix + "_" + Key
#                     )
#                     self.FormulaDict[Key] = self.__PreviousFormula
#                     CurrentModel = LinearModel(self.FormulaDict[Key], self.DataFrame)
#                     self.ANOVA_table.loc[Key, "Sum of Squares"] = np.sum(
#                         np.power(
#                             CurrentModel.FittedValues - self.__PreviousFittedValues, 2
#                         )
#                     )
#                     self.ANOVA_table.loc[Key, "df"] = (
#                         CurrentModel.N - CurrentModel.Df - 1 - self.__PreviousDf
#                     )
#                     self.ANOVA_table.loc[Key, "Mean Square"] = (
#                         self.ANOVA_table.loc[Key, "Sum of Squares"]
#                         / self.ANOVA_table.loc[Key, "df"]
#                     )
#                     self.ANOVA_table.loc[Key, "F Stat"] = self.ANOVA_table.loc[
#                         Key, "Mean Square"
#                     ]

#                     self.__PreviousDf = CurrentModel.N - CurrentModel.Df - 1
#                     assert self.__PreviousDf > 0, "p > n issue happened"
#                     self.__PreviousFittedValues = CurrentModel.FittedValues
#                     self.__InteractionPrefix = (
#                         self.__InteractionPrefix + self.__Prefix + "_" + Key + ":"
#                     )
#                     self.fit(Values)

#                 elif Values == None:
#                     self.FormulaDict[Key] = (
#                         self.__PreviousFormula
#                         + " + "
#                         + self.__InteractionPrefix
#                         + self.__Prefix
#                         + "_"
#                         + Key
#                     )
#                     CurrentModel = LinearModel(self.FormulaDict[Key], self.DataFrame)
#                     self.ANOVA_table.loc[Key, "Sum of Squares"] = np.sum(
#                         np.power(
#                             CurrentModel.FittedValues - self.__PreviousFittedValues, 2
#                         )
#                     )
#                     self.ANOVA_table.loc[Key, "df"] = (
#                         CurrentModel.N - CurrentModel.Df - 1 - self.__PreviousDf
#                     )
#                     assert self.ANOVA_table.loc[Key, "df"] > 0, "p > n issue happened"
#                     self.ANOVA_table.loc[Key, "Mean Square"] = (
#                         self.ANOVA_table.loc[Key, "Sum of Squares"]
#                         / self.ANOVA_table.loc[Key, "df"]
#                     )
#                     self.ANOVA_table.loc[Key, "F Stat"] = self.ANOVA_table.loc[
#                         Key, "Mean Square"
#                     ]

#                     self.ANOVA_table.loc["Residual", "df"] = CurrentModel.Df
#                     self.ANOVA_table.loc["Residual", "Sum of Squares"] = np.sum(
#                         np.power(CurrentModel.Residuals, 2)
#                     )
#                     self.ANOVA_table.loc["Residual", "Mean Square"] = (
#                         self.ANOVA_table.loc["Residual", "Sum of Squares"]
#                         / self.ANOVA_table.loc["Residual", "df"]
#                     )
#                     self.ANOVA_table.loc[:, "F Stat"] /= self.ANOVA_table.loc[
#                         "Residual", "Mean Square"
#                     ]
#                     self.ANOVA_table.loc["Total", "df"] = CurrentModel.N - 1
#                     self.ANOVA_table.loc["Total", "Sum of Squares"] = np.sum(
#                         np.power(
#                             self.DataFrame["_".join(list(self.Y.items())[0])]
#                             - np.mean(
#                                 self.DataFrame["_".join(list(self.Y.items())[0])]
#                             ),
#                             2,
#                         )
#                     )

#             else:
#                 self.__Prefix += Key
#                 self.fit(Values)

#         return None

#     def _get_ANOVA_summary_table(self):
#         for Key in self.ANOVA_table.index:
#             self.ANOVA_table.loc[Key, "P-value"] = 1 - f.cdf(
#                 self.ANOVA_table.loc[Key, "F Stat"],
#                 self.ANOVA_table.loc[Key, "df"],
#                 self.ANOVA_table.loc["Residual", "df"],
#             )
#             if self.ANOVA_table.loc[Key, "P-value"] < 0.001:
#                 self.ANOVA_table.loc[Key, "    "] = "***"
#             elif self.ANOVA_table.loc[Key, "P-value"] < 0.01:
#                 self.ANOVA_table.loc[Key, "    "] = "**"
#             elif self.ANOVA_table.loc[Key, "P-value"] < 0.05:
#                 self.ANOVA_table.loc[Key, "    "] = "*"
#             elif self.ANOVA_table.loc[Key, "P-value"] < 0.1:
#                 self.ANOVA_table.loc[Key, "    "] = "."
#             else:
#                 self.ANOVA_table.loc[Key, "    "] = ""
#         print("      Nested ANOVA Table\n")
#         print(
#             "Response : {}, Number of Data : {}".format(self.Y, self.DataFrame.shape[0])
#         )
#         for k, v in self.FormulaDict.items():
#             print(v)
#         print(self.ANOVA_table)
#         print("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")

#         return None


# import os

# import numpy as np
# import pandas as pd
# from scipy import stats

# from ml_models.linear_models.models.old_linear_model import LinearModel


# # %%
# class Seq_ANOVA:
#     """
#     This object using sequential ANOVA analysis to find significant factors, and it'll return sequential ANOVA summary table.
#     """

#     def __init__(self, Y, DataFrame):
#         """
#         Parameters
#         -----
#         Y : dict
#             1st input argument. Protected
#         DataFrame : DataFrame
#             2rd input argument. Public
#         """
#         pd.options.display.float_format = "{:,.4f}".format
#         self.__Y = "_".join(list(Y.items())[0])
#         self.__InfoNest = []
#         ####
#         BasicFormula = self.__Y + " ~ 1"
#         ####
#         self.DataFrame = DataFrame
#         self.__DataFrame = self.DataFrame.copy()
#         self.__DataFrame.columns = self.__DataFrame.columns.map("_".join)
#         self.__DegOfFree = dict()
#         self.__RSSDegOfFree = dict()
#         self.__SumOfSquare = dict()
#         self.__FittedValues = dict()
#         self.__RegSumOfSquare = dict()
#         self.Formula = dict()
#         self.__Formula = dict()
#         self.__SignificantStar = dict()
#         self.PValue = dict()
#         self.FValue = dict()

#         assert type(self.__Y) is str
#         BasicModel = LinearModel(BasicFormula, DataFrame=self.__DataFrame)
#         # model info.
#         self.__DegOfFree["Intercept"] = int(0)
#         self.__RSSDegOfFree["Intercept"] = int(self.DataFrame.shape[0] - 1)
#         self.__FittedValues["Intercept"] = BasicModel.FittedValues
#         self.__SumOfSquare["Intercept"] = " "
#         self.Formula["Intercept"] = BasicFormula
#         self.__Formula["Intercept"] = BasicFormula
#         self.__SignificantStar["Intercept"] = " "
#         self.__RegSumOfSquare["Intercept"] = float(np.sum(BasicModel.Residuals**2))
#         self.FValue["Intercept"] = " "
#         self.PValue["Intercept"] = " "
#         # info. name record
#         self.__PreviousInfoName = "Intercept"
#         self.__PreviousFormula = BasicFormula

#     def updatemodel(self, InfoName, X):
#         """
#         This updatemodel is to update the ANOVA model and original information when we add another formula.

#         Parameters
#         -----
#         InfoName : str
#             1th input argument. Public
#         X : dict
#             2nd input argument. Public
#         """
#         if type(X) is dict:
#             Key, Values = list(X.items())[0]
#             self.__InfoNest.append(Values)
#             self.Formula[InfoName] = self.Formula[self.__PreviousInfoName] + " + " + Key
#             self.__Formula[InfoName] = self.__Formula[
#                 self.__PreviousInfoName
#             ] + "".join(
#                 [
#                     " + " + key + "_" + values
#                     for key, values in zip([Key] * len(Values), Values)
#                 ]
#             )
#         else:
#             print("Type of X must be dict")
#             return None
#         updatedModel = LinearModel(self.__Formula[InfoName], DataFrame=self.__DataFrame)
#         self.__DegOfFree[InfoName] = int(
#             len(updatedModel.BetaHat) - sum(self.__DegOfFree.values()) - 1
#         )
#         self.__RSSDegOfFree[InfoName] = int(
#             self.__RSSDegOfFree[self.__PreviousInfoName] - self.__DegOfFree[InfoName]
#         )
#         self.__FittedValues[InfoName] = updatedModel.FittedValues
#         self.__SumOfSquare[InfoName] = float(
#             sum(
#                 (
#                     updatedModel.FittedValues
#                     - self.__FittedValues[self.__PreviousInfoName]
#                 )
#                 ** 2
#             )
#         )
#         self.__RegSumOfSquare[InfoName] = float(sum(updatedModel.Residuals**2))
#         self.FValue[InfoName] = float(
#             (
#                 self.__RegSumOfSquare[self.__PreviousInfoName]
#                 - self.__RegSumOfSquare[InfoName]
#             )
#             / self.__DegOfFree[InfoName]
#         )
#         self.__PreviousInfoName = InfoName

#     def finish(self):
#         """
#         This finish is to construct the sequential ANOVA summary table, and it'll return a sequential ANOVA summary table.
#         """
#         # calculate F value and P value
#         SecDegOfFree = self.DataFrame.shape[0] - sum(self.__DegOfFree.values()) - 1
#         DivisionScale = self.__RegSumOfSquare[self.__PreviousInfoName] / SecDegOfFree
#         for InfoName in self.FValue.keys():
#             if InfoName != "Intercept":
#                 self.FValue[InfoName] /= DivisionScale
#                 self.PValue[InfoName] = 1 - stats.f.cdf(
#                     self.FValue[InfoName], self.__DegOfFree[InfoName], SecDegOfFree
#                 )
#                 if self.PValue[InfoName] < 0.001:
#                     self.__SignificantStar[InfoName] = "***"
#                 elif self.PValue[InfoName] < 0.01:
#                     self.__SignificantStar[InfoName] = "**"
#                 elif self.PValue[InfoName] < 0.05:
#                     self.__SignificantStar[InfoName] = "*"
#                 elif self.PValue[InfoName] < 0.1:
#                     self.__SignificantStar[InfoName] = "."
#                 else:
#                     self.__SignificantStar[InfoName] = ""
#         self.AnovaTable = pd.concat(
#             [
#                 pd.DataFrame.from_dict(self.__RSSDegOfFree, orient="index"),
#                 pd.DataFrame.from_dict(self.__RegSumOfSquare, orient="index"),
#                 pd.DataFrame.from_dict(self.__DegOfFree, orient="index"),
#                 pd.DataFrame.from_dict(self.__SumOfSquare, orient="index"),
#                 pd.DataFrame.from_dict(self.FValue, orient="index"),
#                 pd.DataFrame.from_dict(self.PValue, orient="index"),
#                 pd.DataFrame.from_dict(self.__SignificantStar, orient="index"),
#             ],
#             axis=1,
#         )
#         self.AnovaTable.columns = [
#             "Reg Df",
#             "Reg Sum of Sq",
#             "Df",
#             "Sum of Sq",
#             "F",
#             "Pr(>F)",
#             " ",
#         ]

#         return self.AnovaTable

#     def _get_ANOVA_summary_table(self):
#         """
#         This _get_ANOVA_summary_table is to print the sequential ANOVA summary table.
#         """
#         for key, pvalue in zip(self.PValue.keys(), self.PValue.values()):
#             if key != "Intercept":
#                 self.PValue[key] = "{:.3g}".format(pvalue)
#         self.__DegOfFree["Intercept"] = " "

#         print("      Nested Sequential ANOVA Table\n")
#         print(
#             "Response : {}, Number of Data : {}".format(
#                 self.__Y, self.DataFrame.shape[0]
#             ),
#             end=", ",
#         )
#         print("Information in Nest: ", end="")
#         print(*np.unique(self.__InfoNest), sep=", ", end="\n\n")
#         i = 0
#         for InfoName in self.FValue.keys():
#             print(str(i + 1) + ". Model " + str(i) + " : " + self.Formula[InfoName])
#             i += 1
#         print(self.AnovaTable)
#         print("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")

#         return None


# # %%
# # simulation : nested sequential ANOVA
# if __name__ == "__main__":
#     # ANOVA assumptions :
#     # nomality, epsilon_ijk iid~ N(0, sigma); for i = 1,...,n_i, j = 1,...,n_ij, k = 1,...,n_ijk
#     import warnings

#     warnings.filterwarnings("ignore")

#     ## Data generation ##
#     seed = np.random.RandomState(1122)
#     df = pd.DataFrame(
#         np.array(
#             [
#                 seed.choice(["AKCLA100", "AKCLA200", "AKCLA300", "AKCLA400"], 2000),
#                 seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
#                 seed.choice(["AKSPT100", "AKSPT200", "AKSPT300", "AKSPT400"], 2000),
#                 seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
#                 seed.choice(["AKIEX100", "AKIEX200", "AKIEX300", "AKIEX400"], 2000),
#                 seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
#                 seed.choice(["AKWMA100", "AKWMA200", "AKWMA300", "AKWMA400"], 2000),
#                 seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
#                 seed.choice(["AKSTO100", "AKSTO200", "AKSTO300", "AKSTO400"], 2000),
#                 seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
#             ]
#         ).T,
#         columns=pd.MultiIndex.from_product(
#             [["GL_CLA", "GL_SPT", "GL_IEX", "GL_WMA", "GL_STO"], ["EQP", "CHAMBER"]],
#             names=["OP_ID", "OP_Levels"],
#         ),
#     )
#     df[("YLD", "TOS1")] = seed.randn(2000)
#     # AKIEX300 has a positive effect to TOS YLD
#     df.loc[df[("GL_IEX", "EQP")] == "AKIEX300", ("YLD", "TOS1")] += 0.2
#     # AKCLA100, Chamber 2 has a positive effect to TOS YLD
#     df.loc[
#         (df[("GL_CLA",)] == ("AKCLA100", "Chamber_2")).sum(axis=1) == 2, ("YLD", "TOS1")
#     ] += 0.2

#     ## Fit ANOVA ##
#     anova = Seq_ANOVA(Y={"YLD": "TOS1"}, DataFrame=df)
#     anova.updatemodel("GL_CLA info", {"GL_CLA": {"EQP": None, "CHAMBER": None}})
#     anova.updatemodel("GL_SPT info", {"GL_SPT": {"EQP": None, "CHAMBER": None}})
#     anova.updatemodel("GL_IEX info", {"GL_IEX": {"EQP": None, "CHAMBER": None}})
#     anova.updatemodel("GL_WMA info", {"GL_WMA": {"EQP": None, "CHAMBER": None}})
#     anova.updatemodel("GL_STO info", {"GL_STO": {"EQP": None, "CHAMBER": None}})
#     anova.finish()
#     anova._get_ANOVA_summary_table()
# %%
