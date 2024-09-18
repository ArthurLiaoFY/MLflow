# %%
from configparser import ConfigParser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_projects.curve_classify.load_data import load_data
from ml_projects.curve_classify.plot_fn import plot_curve

# %%

config = ConfigParser()
config.read("curve_classify.ini")
# %%
curve, cum_curve, label = load_data(
    data_file_path=config["curve_classify"]["data_file_path"]
)
# %%
np.unique(
    label["test_result"],
    return_counts=True,
)

# %%
plot_curve(
    curve=cum_curve,
    label=label,
    plot_file_path=config["curve_classify"]["plot_file_path"],
)

# %%
train_idx, val_idx = train_test_split(
    label.index,
    train_size=0.8,
    random_state=1122,
    shuffle=True,
    stratify=label["test_result"],
)

# %%
train_x = cum_curve.iloc[train_idx, :]
val_x = cum_curve.iloc[val_idx, :]
train_y = label.iloc[train_idx, :]
val_y = label.iloc[val_idx, :]
# %%
