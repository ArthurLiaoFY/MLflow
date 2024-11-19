import numpy as np
import pandas as pd


class GetDummies:
    def __init__(self, dismiss_amount: int = 10, dismiss_name: str = "Others"):
        self.dummy_map = {}
        self.dismiss_amount = dismiss_amount
        self.dismiss_name = dismiss_name

    def get_dummies(
        self,
        df: pd.DataFrame,
        colname: str,
    ) -> np.ndarray:
        if self.dummy_map.get(colname) == None:
            uniq_cat, cat_cnt = np.unique(
                df[colname].fillna(self.dismiss_name).astype(str),
                return_counts=True,
                equal_nan=self.dismiss_name,
            )
            set_to_other_cat = uniq_cat[cat_cnt <= self.dismiss_amount]
            if len(set_to_other_cat) == 0:
                set_to_other_cat = uniq_cat[np.argmin(cat_cnt)]

            raw_dummies = (
                pd.get_dummies(
                    data=df[colname].apply(
                        lambda x: self.dismiss_name if x in set_to_other_cat else x
                    ),
                    drop_first=False,
                )
                .astype(int)
                .drop(columns=self.dismiss_name)
            )
            self.dummy_map[colname] = {
                **{
                    df.loc[idx, colname]: dummy_series.to_numpy()[np.newaxis, :]
                    for idx, dummy_series in raw_dummies.drop_duplicates().iterrows()
                    if df.loc[idx, colname] not in set_to_other_cat
                },
                **{
                    self.dismiss_name: np.zeros(shape=raw_dummies.shape[-1], dtype=int)[
                        np.newaxis, :
                    ]
                },
            }
            return raw_dummies.to_numpy()
        else:

            return np.concatenate(
                [
                    self.dummy_map[colname].get(
                        cat, self.dummy_map[colname][self.dismiss_name]
                    )
                    for cat in df[colname]
                ],
                axis=0,
            )


class Standardize:
    @staticmethod
    def sigmoid(
        df: pd.DataFrame,
        colname: str,
    ) -> np.ndarray:
        return 1 / (1 + np.exp(-df[colname]))

    @staticmethod
    def softmax(
        df: pd.DataFrame,
        colname: str,
    ) -> np.ndarray:
        exp_values = np.exp(df[colname] - np.max(df[colname]))
        return exp_values / exp_values.sum()


class ToType:
    @staticmethod
    def _to_type(df: pd.DataFrame, colname: str, type: type):
        return df.get(colname).astype(type).to_numpy()[:, np.newaxis]

    def to_int(
        self,
        df: pd.DataFrame,
        colname: str,
    ):
        return self._to_type(df, colname, int)

    def to_str(
        self,
        df: pd.DataFrame,
        colname: str,
    ):
        return self._to_type(df, colname, str)

    def to_float(
        self,
        df: pd.DataFrame,
        colname: str,
    ):
        return self._to_type(df, colname, float)
