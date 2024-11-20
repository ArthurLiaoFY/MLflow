import numpy as np
import pandas as pd

from ml_models.common.transform_table_data import GetDummies, Standardize, ToType


class TransformData(GetDummies, Standardize, ToType):
    def __init__(self, dismiss_amount: int = 10, dismiss_name: str = "Others"):
        super().__init__(dismiss_amount, dismiss_name)

    def fit_transform(self, df: pd.DataFrame, inference: bool = False) -> np.ndarray:
        transformed_data = [
            self.get_dummies(df=df, colname="城市"),
            self.get_dummies(df=df, colname="鄉鎮市區"),
            self.to_int(df=df, colname="土地數量"),
            self.to_int(df=df, colname="建物數量"),
            self.to_int(df=df, colname="車位數量"),
            self.get_dummies(df=df, colname="建物型態"),
            self.get_dummies(df=df, colname="記錄季度"),
            self.to_float(df=df, colname="土地移轉總面積平方公尺"),
            self.to_float(df=df, colname="建物移轉總面積平方公尺"),
            self.to_float(df=df, colname="車位移轉總面積平方公尺"),
            self.to_int(df=df, colname="建物現況格局-房"),
            self.to_int(df=df, colname="建物現況格局-廳"),
            self.to_int(df=df, colname="建物現況格局-衛"),
            self.to_int(df=df, colname="建物現況格局-隔間"),
        ]

        if not inference:
            transformed_data.append(
                self.to_float(df=df, colname="總價元"),
            )

        return np.concatenate(transformed_data, axis=1)
