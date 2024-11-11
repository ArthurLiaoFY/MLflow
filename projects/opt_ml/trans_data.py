import numpy as np
import pandas as pd

from models.common.transform_table_data import GetDummies


class TransformData(GetDummies):
    def __init__(self, dismiss_amount: int = 10, dismiss_name: str = "Others"):
        super().__init__(dismiss_amount, dismiss_name)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.concatenate(
            [
                self.get_dummies(df=df, colname="城市"),
                self.get_dummies(df=df, colname="鄉鎮市區"),
                pd.DataFrame.from_dict(
                    {
                        idx: dict(zip(key_l, value_l))
                        for idx, key_l, value_l in zip(
                            df.index,
                            df.loc[:, "交易筆棟數"].str.findall(r"[^\d]+").to_numpy(),
                            df.loc[:, "交易筆棟數"].str.findall(r"\d+").to_numpy(),
                        )
                    },
                    orient="index",
                )
                .add_suffix("數量")
                .astype(int)
                .to_numpy(),
                self.get_dummies(df=df, colname="建物型態"),
                self.get_dummies(df=df, colname="記錄季度"),
                df.get("土地移轉總面積平方公尺")
                .astype(float)
                .to_numpy()[:, np.newaxis],
                df.get("建物移轉總面積平方公尺")
                .astype(float)
                .to_numpy()[:, np.newaxis],
                df.get("車位移轉總面積平方公尺")
                .astype(float)
                .to_numpy()[:, np.newaxis],
                df.get("建物現況格局-房").astype(int).to_numpy()[:, np.newaxis],
                df.get("建物現況格局-廳").astype(int).to_numpy()[:, np.newaxis],
                df.get("建物現況格局-衛").astype(int).to_numpy()[:, np.newaxis],
                df.get("建物現況格局-隔間")
                .apply(lambda x: 1 if x == "有" else 0)
                .astype(int)
                .to_numpy()[:, np.newaxis],
                ##############
                df.get("總價元").astype(float).to_numpy()[:, np.newaxis],
            ],
            axis=1,
        )
