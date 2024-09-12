import os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir('D:/UserData/ArthurLiao/golden_path_phase2/')
from LinearModel import LinearModel


class Nested_ANOVA:
    """
    This object using Nested ANOVA analysis to find significant factors, and it'll return Nested ANOVA summary table.
    """

    def __init__(self, Y, X, DataFrame):
        """
        Parameters
        -----
        Y : dict
            1st input argument. Protected
        X : dict
            2st input argument. Protected
        DataFrame : DataFrame
            3rd input argument. Public
        """
        self.Y = Y
        assert type(self.Y) == dict, 'Y must be a dict'
        self.X = X
        assert type(self.X) == dict, 'X must be a dict'
        self.DataFrame = DataFrame.copy()
        assert type(self.DataFrame) == pd.core.frame.DataFrame
        assert tuple(self.Y.items())[0] in self.DataFrame.columns, 'the dataframe you import must contain {}'.format(
            self.Y)
        self.DataFrame.columns = self.DataFrame.columns.map('_'.join)
        assert self.DataFrame['_'.join(
            list(self.Y.items())[0])].dtype != object, 'The data type of {} must be \'float\' or \'int\''.format(self.Y)
        self.FormulaDict = dict()
        self.__ColumnList = ['Sum of Squares', 'df', 'Mean Square', 'F Stat', 'P-value', '    ']
        self.ANOVA_table = pd.DataFrame(columns=self.__ColumnList)
        self.__Prefix = ''
        self.__InteractionPrefix = ''
        self.__PreviousFormula = '_'.join(list(self.Y.items())[0]) + ' ~ 1'
        self.__PreviousFittedValues = LinearModel(self.__PreviousFormula, self.DataFrame).FittedValues
        self.__PreviousDf = 0
        self.fit(self.X)

    def fit(self, Dict):
        for Key, Values in Dict.items():
            if self.__Prefix != '':
                self.ANOVA_table = self.ANOVA_table.append(
                    pd.DataFrame(index=[Key], columns=self.__ColumnList))
                if isinstance(Values, dict):
                    self.__PreviousFormula += ' + ' + self.__InteractionPrefix + self.__Prefix + '_' + Key
                    self.FormulaDict[Key] = self.__PreviousFormula
                    CurrentModel = LinearModel(self.FormulaDict[Key], self.DataFrame)
                    self.ANOVA_table.loc[Key, 'Sum of Squares'] = \
                        np.sum(np.power(CurrentModel.FittedValues - self.__PreviousFittedValues, 2))
                    self.ANOVA_table.loc[Key, 'df'] = CurrentModel.N - CurrentModel.Df - 1 - self.__PreviousDf
                    self.ANOVA_table.loc[Key, 'Mean Square'] = self.ANOVA_table.loc[Key, 'Sum of Squares'] / \
                                                               self.ANOVA_table.loc[Key, 'df']
                    self.ANOVA_table.loc[Key, 'F Stat'] = self.ANOVA_table.loc[Key, 'Mean Square']

                    self.__PreviousDf = CurrentModel.N - CurrentModel.Df - 1
                    assert self.__PreviousDf > 0, 'p > n issue happened'
                    self.__PreviousFittedValues = CurrentModel.FittedValues
                    self.__InteractionPrefix = self.__InteractionPrefix + self.__Prefix + '_' + Key + ':'
                    self.fit(Values)

                elif Values == None:
                    self.FormulaDict[
                        Key] = self.__PreviousFormula + ' + ' + self.__InteractionPrefix + self.__Prefix + '_' + Key
                    CurrentModel = LinearModel(self.FormulaDict[Key], self.DataFrame)
                    self.ANOVA_table.loc[Key, 'Sum of Squares'] = \
                        np.sum(np.power(CurrentModel.FittedValues - self.__PreviousFittedValues, 2))
                    self.ANOVA_table.loc[Key, 'df'] = CurrentModel.N - CurrentModel.Df - 1 - self.__PreviousDf
                    assert self.ANOVA_table.loc[Key, 'df'] > 0, 'p > n issue happened'
                    self.ANOVA_table.loc[Key, 'Mean Square'] = self.ANOVA_table.loc[Key, 'Sum of Squares'] / \
                                                               self.ANOVA_table.loc[Key, 'df']
                    self.ANOVA_table.loc[Key, 'F Stat'] = self.ANOVA_table.loc[Key, 'Mean Square']

                    self.ANOVA_table.loc['Residual', 'df'] = CurrentModel.Df
                    self.ANOVA_table.loc['Residual', 'Sum of Squares'] = np.sum(np.power(CurrentModel.Residuals, 2))
                    self.ANOVA_table.loc['Residual', 'Mean Square'] = self.ANOVA_table.loc[
                                                                          'Residual', 'Sum of Squares'] / \
                                                                      self.ANOVA_table.loc['Residual', 'df']
                    self.ANOVA_table.loc[:, 'F Stat'] /= self.ANOVA_table.loc['Residual', 'Mean Square']
                    self.ANOVA_table.loc['Total', 'df'] = CurrentModel.N - 1
                    self.ANOVA_table.loc['Total', 'Sum of Squares'] = \
                        np.sum(np.power(self.DataFrame['_'.join(list(self.Y.items())[0])] -
                                        np.mean(self.DataFrame['_'.join(list(self.Y.items())[0])]), 2))

            else:
                self.__Prefix += Key
                self.fit(Values)

        return None

    def _get_ANOVA_summary_table(self):
        for Key in self.ANOVA_table.index:
            self.ANOVA_table.loc[Key, 'P-value'] = 1 - stats.f.cdf(self.ANOVA_table.loc[Key, 'F Stat'],
                                                                   self.ANOVA_table.loc[Key, 'df'],
                                                                   self.ANOVA_table.loc['Residual', 'df'])
            if self.ANOVA_table.loc[Key, 'P-value'] < 0.001:
                self.ANOVA_table.loc[Key, '    '] = '***'
            elif self.ANOVA_table.loc[Key, 'P-value'] < 0.01:
                self.ANOVA_table.loc[Key, '    '] = '**'
            elif self.ANOVA_table.loc[Key, 'P-value'] < 0.05:
                self.ANOVA_table.loc[Key, '    '] = '*'
            elif self.ANOVA_table.loc[Key, 'P-value'] < 0.1:
                self.ANOVA_table.loc[Key, '    '] = '.'
            else:
                self.ANOVA_table.loc[Key, '    '] = ''
        print('      Nested ANOVA Table\n')
        print('Response : {}, Number of Data : {}'.format(self.Y, self.DataFrame.shape[0]))
        for k, v in self.FormulaDict.items():
            print(v)
        print(self.ANOVA_table)
        print('---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1')

        return None
