# %%
from configparser import ConfigParser

import numpy as np

from projects.curve_classify.load_data import load_data
from projects.curve_classify.plot_fns import plot_curve

# %%

config = ConfigParser()
config.read("projects/curve_classify/curve_classify.ini")
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
