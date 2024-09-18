import pandas as pd


def load_data() -> tuple[pd.DataFrame]:
    curve = pd.read_table(
        "secom/secom_data.txt",
        sep=" ",
        header=None,
    )
    curve.fillna(0, inplace=True)
    cum_curve = curve.cumsum(axis=1, skipna=True)
    label = pd.read_table(
        "secom/secom_labels.txt",
        sep=" ",
        header=None,
    )
    label.columns = ["test_result", "test_time"]

    return curve, cum_curve, label
