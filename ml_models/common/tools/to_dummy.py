import numpy as np
import pandas as pd


def sum_coding_matrix(
    data: np.ndarray,
    prefix: str = "",
    prefix_sep: str = "_",
    baseline: int | str | None = None,
) -> np.ndarray:
    data = data.values if isinstance(data, pd.Series) else data
    unique_cats = np.sort(np.unique(data).astype(str))

    baseline = (
        str(baseline)
        if baseline is not None and str(baseline) in unique_cats
        else unique_cats[-1]
    )

    sum_coding_columns = [
        f"{prefix}{prefix_sep}{cat}" for cat in unique_cats if cat != baseline
    ]

    sum_coding = np.zeros((len(data), len(sum_coding_columns)))

    baseline_mask = data == baseline
    sum_coding[baseline_mask, :] = -1

    for i, cat in enumerate(unique_cats):
        if cat != baseline:
            sum_coding[
                data == cat, i - (i > np.where(unique_cats == baseline)[0][0])
            ] = 1

    return pd.DataFrame(sum_coding, columns=sum_coding_columns)
