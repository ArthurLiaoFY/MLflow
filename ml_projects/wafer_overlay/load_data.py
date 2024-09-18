import numpy as np
import pandas as pd
import seaborn as sns

from common.plot_fn import plot_wafer


def load_data(data_file_path: str, drop_columns: list[str] | None = None):
    if not drop_columns:

        df = pd.read_csv(f"{data_file_path}/data.csv", index_col=0).drop_duplicates()

    else:
        df = (
            pd.read_csv(f"{data_file_path}/data.csv", index_col=0)
            .drop(columns=drop_columns)
            .drop_duplicates()
        )
    df["WaferStartTime"] = pd.to_datetime(df["WaferStartTime"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("WaferStartTime").reset_index(drop=True)
    return df
