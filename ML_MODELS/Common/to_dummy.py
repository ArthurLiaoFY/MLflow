import numpy as np
import pandas as pd


def sum_coding_matrix(
    data: np.ndarray,
    prefix: str = "",
    prefix_sep: str = "_",
    baseline: int | str | None = None,
) -> np.ndarray:
    # 如果 data 是 pd.Series，则提取其值
    data = data.values if isinstance(data, pd.Series) else data
    unique_cats = np.sort(np.unique(data).astype(str))  # 使用 numpy 排序和唯一值处理

    # 设置 baseline，如果未提供或 baseline 不在 unique_cats 中，使用最后一个类别作为 baseline
    baseline = (
        str(baseline)
        if baseline is not None and str(baseline) in unique_cats
        else unique_cats[-1]
    )

    # 创建编码列名，排除 baseline 类别
    sum_coding_columns = [
        f"{prefix}{prefix_sep}{cat}" for cat in unique_cats if cat != baseline
    ]

    # 初始化编码矩阵
    sum_coding = np.zeros((len(data), len(sum_coding_columns)))

    # 为 baseline 类别设置 -1
    baseline_mask = data == baseline
    sum_coding[baseline_mask, :] = -1

    # 为非 baseline 类别设置 1，利用 numpy 的广播机制
    for i, cat in enumerate(unique_cats):
        if cat != baseline:
            sum_coding[
                data == cat, i - (i > np.where(unique_cats == baseline)[0][0])
            ] = 1

    # 返回 pandas DataFrame，带有指定的列名
    return pd.DataFrame(sum_coding, columns=sum_coding_columns)
