import os

import numpy as np
import pandas as pd
from scipy import stats

from models.linear_models.models.linear_model import LinearModel


# %%
class Seq_ANOVA:
    """
    This object using sequential ANOVA analysis to find significant factors, and it'll return sequential ANOVA summary table.
    """

    def __init__(self, Y, DataFrame):
        """
        Parameters
        -----
        Y : dict
            1st input argument. Protected
        DataFrame : DataFrame
            2rd input argument. Public
        """
        pd.options.display.float_format = "{:,.4f}".format
        self.__Y = "_".join(list(Y.items())[0])
        self.__InfoNest = []
        ####
        BasicFormula = self.__Y + " ~ 1"
        ####
        self.DataFrame = DataFrame
        self.__DataFrame = self.DataFrame.copy()
        self.__DataFrame.columns = self.__DataFrame.columns.map("_".join)
        self.__DegOfFree = dict()
        self.__RSSDegOfFree = dict()
        self.__SumOfSquare = dict()
        self.__FittedValues = dict()
        self.__RegSumOfSquare = dict()
        self.Formula = dict()
        self.__Formula = dict()
        self.__SignificantStar = dict()
        self.PValue = dict()
        self.FValue = dict()

        assert type(self.__Y) is str
        BasicModel = LinearModel(BasicFormula, DataFrame=self.__DataFrame)
        # model info.
        self.__DegOfFree["Intercept"] = int(0)
        self.__RSSDegOfFree["Intercept"] = int(self.DataFrame.shape[0] - 1)
        self.__FittedValues["Intercept"] = BasicModel.FittedValues
        self.__SumOfSquare["Intercept"] = " "
        self.Formula["Intercept"] = BasicFormula
        self.__Formula["Intercept"] = BasicFormula
        self.__SignificantStar["Intercept"] = " "
        self.__RegSumOfSquare["Intercept"] = float(np.sum(BasicModel.Residuals**2))
        self.FValue["Intercept"] = " "
        self.PValue["Intercept"] = " "
        # info. name record
        self.__PreviousInfoName = "Intercept"
        self.__PreviousFormula = BasicFormula

    def updatemodel(self, InfoName, X):
        """
        This updatemodel is to update the ANOVA model and original information when we add another formula.

        Parameters
        -----
        InfoName : str
            1th input argument. Public
        X : dict
            2nd input argument. Public
        """
        if type(X) is dict:
            Key, Values = list(X.items())[0]
            self.__InfoNest.append(Values)
            self.Formula[InfoName] = self.Formula[self.__PreviousInfoName] + " + " + Key
            self.__Formula[InfoName] = self.__Formula[
                self.__PreviousInfoName
            ] + "".join(
                [
                    " + " + key + "_" + values
                    for key, values in zip([Key] * len(Values), Values)
                ]
            )
        else:
            print("Type of X must be dict")
            return None
        updatedModel = LinearModel(self.__Formula[InfoName], DataFrame=self.__DataFrame)
        self.__DegOfFree[InfoName] = int(
            len(updatedModel.BetaHat) - sum(self.__DegOfFree.values()) - 1
        )
        self.__RSSDegOfFree[InfoName] = int(
            self.__RSSDegOfFree[self.__PreviousInfoName] - self.__DegOfFree[InfoName]
        )
        self.__FittedValues[InfoName] = updatedModel.FittedValues
        self.__SumOfSquare[InfoName] = float(
            sum(
                (
                    updatedModel.FittedValues
                    - self.__FittedValues[self.__PreviousInfoName]
                )
                ** 2
            )
        )
        self.__RegSumOfSquare[InfoName] = float(sum(updatedModel.Residuals**2))
        self.FValue[InfoName] = float(
            (
                self.__RegSumOfSquare[self.__PreviousInfoName]
                - self.__RegSumOfSquare[InfoName]
            )
            / self.__DegOfFree[InfoName]
        )
        self.__PreviousInfoName = InfoName

    def finish(self):
        """
        This finish is to construct the sequential ANOVA summary table, and it'll return a sequential ANOVA summary table.
        """
        # calculate F value and P value
        SecDegOfFree = self.DataFrame.shape[0] - sum(self.__DegOfFree.values()) - 1
        DivisionScale = self.__RegSumOfSquare[self.__PreviousInfoName] / SecDegOfFree
        for InfoName in self.FValue.keys():
            if InfoName != "Intercept":
                self.FValue[InfoName] /= DivisionScale
                self.PValue[InfoName] = 1 - stats.f.cdf(
                    self.FValue[InfoName], self.__DegOfFree[InfoName], SecDegOfFree
                )
                if self.PValue[InfoName] < 0.001:
                    self.__SignificantStar[InfoName] = "***"
                elif self.PValue[InfoName] < 0.01:
                    self.__SignificantStar[InfoName] = "**"
                elif self.PValue[InfoName] < 0.05:
                    self.__SignificantStar[InfoName] = "*"
                elif self.PValue[InfoName] < 0.1:
                    self.__SignificantStar[InfoName] = "."
                else:
                    self.__SignificantStar[InfoName] = ""
        self.AnovaTable = pd.concat(
            [
                pd.DataFrame.from_dict(self.__RSSDegOfFree, orient="index"),
                pd.DataFrame.from_dict(self.__RegSumOfSquare, orient="index"),
                pd.DataFrame.from_dict(self.__DegOfFree, orient="index"),
                pd.DataFrame.from_dict(self.__SumOfSquare, orient="index"),
                pd.DataFrame.from_dict(self.FValue, orient="index"),
                pd.DataFrame.from_dict(self.PValue, orient="index"),
                pd.DataFrame.from_dict(self.__SignificantStar, orient="index"),
            ],
            axis=1,
        )
        self.AnovaTable.columns = [
            "Reg Df",
            "Reg Sum of Sq",
            "Df",
            "Sum of Sq",
            "F",
            "Pr(>F)",
            " ",
        ]

        return self.AnovaTable

    def _get_ANOVA_summary_table(self):
        """
        This _get_ANOVA_summary_table is to print the sequential ANOVA summary table.
        """
        for key, pvalue in zip(self.PValue.keys(), self.PValue.values()):
            if key != "Intercept":
                self.PValue[key] = "{:.3g}".format(pvalue)
        self.__DegOfFree["Intercept"] = " "

        print("      Nested Sequential ANOVA Table\n")
        print(
            "Response : {}, Number of Data : {}".format(
                self.__Y, self.DataFrame.shape[0]
            ),
            end=", ",
        )
        print("Information in Nest: ", end="")
        print(*np.unique(self.__InfoNest), sep=", ", end="\n\n")
        i = 0
        for InfoName in self.FValue.keys():
            print(str(i + 1) + ". Model " + str(i) + " : " + self.Formula[InfoName])
            i += 1
        print(self.AnovaTable)
        print("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")

        return None


# %%
# simulation : nested sequential ANOVA
if __name__ == "__main__":
    # ANOVA assumptions :
    # nomality, epsilon_ijk iid~ N(0, sigma); for i = 1,...,n_i, j = 1,...,n_ij, k = 1,...,n_ijk
    import warnings

    warnings.filterwarnings("ignore")

    ## Data generation ##
    seed = np.random.RandomState(1122)
    df = pd.DataFrame(
        np.array(
            [
                seed.choice(["AKCLA100", "AKCLA200", "AKCLA300", "AKCLA400"], 2000),
                seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
                seed.choice(["AKSPT100", "AKSPT200", "AKSPT300", "AKSPT400"], 2000),
                seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
                seed.choice(["AKIEX100", "AKIEX200", "AKIEX300", "AKIEX400"], 2000),
                seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
                seed.choice(["AKWMA100", "AKWMA200", "AKWMA300", "AKWMA400"], 2000),
                seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
                seed.choice(["AKSTO100", "AKSTO200", "AKSTO300", "AKSTO400"], 2000),
                seed.choice(["Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4"], 2000),
            ]
        ).T,
        columns=pd.MultiIndex.from_product(
            [["GL_CLA", "GL_SPT", "GL_IEX", "GL_WMA", "GL_STO"], ["EQP", "CHAMBER"]],
            names=["OP_ID", "OP_Levels"],
        ),
    )
    df[("YLD", "TOS1")] = seed.randn(2000)
    # AKIEX300 has a positive effect to TOS YLD
    df.loc[df[("GL_IEX", "EQP")] == "AKIEX300", ("YLD", "TOS1")] += 0.2
    # AKCLA100, Chamber 2 has a positive effect to TOS YLD
    df.loc[
        (df[("GL_CLA",)] == ("AKCLA100", "Chamber_2")).sum(axis=1) == 2, ("YLD", "TOS1")
    ] += 0.2

    ## Fit ANOVA ##
    anova = Seq_ANOVA(Y={"YLD": "TOS1"}, DataFrame=df)
    anova.updatemodel("GL_CLA info", {"GL_CLA": {"EQP": None, "CHAMBER": None}})
    anova.updatemodel("GL_SPT info", {"GL_SPT": {"EQP": None, "CHAMBER": None}})
    anova.updatemodel("GL_IEX info", {"GL_IEX": {"EQP": None, "CHAMBER": None}})
    anova.updatemodel("GL_WMA info", {"GL_WMA": {"EQP": None, "CHAMBER": None}})
    anova.updatemodel("GL_STO info", {"GL_STO": {"EQP": None, "CHAMBER": None}})
    anova.finish()
    anova._get_ANOVA_summary_table()
