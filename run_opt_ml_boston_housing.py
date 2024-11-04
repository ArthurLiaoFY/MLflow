# %%
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from projects.opt_ml.estimate_surface import EstimateSurface
from projects.opt_ml.optimize_response import optimize_f_hat
from projects.opt_ml.plot_fns import plot_obj_surface

config = ConfigParser()
config.read("projects/opt_ml/opt_ml.ini")
config = config["boston_housing"]


train_df = pd.read_csv("./data/boston_housing/train.csv")
train_df = train_df.loc[train_df["2ndFlrSF"] > 0, :]

# %%
x_min = [
    train_df["1stFlrSF"].quantile(float(config.get("lower_quantile"))),
    train_df["2ndFlrSF"].quantile(float(config.get("lower_quantile"))),
]

x_max = [
    train_df["1stFlrSF"].quantile(float(config.get("upper_quantile"))),
    train_df["2ndFlrSF"].quantile(float(config.get("upper_quantile"))),
]
# %%
# plt.plot(train_df["1stFlrSF"], train_df["2ndFlrSF"], "o")
# plt.vlines(
#     x=train_df["1stFlrSF"].quantile(float(config.get("lower_quantile"))),
#     ymin=x_min[1],
#     ymax=x_max[1],
#     colors="red",
#     linestyles="--",
# )
# plt.vlines(
#     x=train_df["1stFlrSF"].quantile(float(config.get("upper_quantile"))),
#     ymin=x_min[1],
#     ymax=x_max[1],
#     colors="red",
#     linestyles="--",
# )

# plt.hlines(
#     y=train_df["2ndFlrSF"].quantile(float(config.get("lower_quantile"))),
#     xmin=x_min[0],
#     xmax=x_max[0],
#     colors="red",
#     linestyles="--",
# )
# plt.hlines(
#     y=train_df["2ndFlrSF"].quantile(float(config.get("upper_quantile"))),
#     xmin=x_min[0],
#     xmax=x_max[0],
#     colors="red",
#     linestyles="--",
# )
# plt.show()

# %%

sm = EstimateSurface(run_id="boston_housing", **config)
sm.fit_surface(X=train_df[["1stFlrSF", "2ndFlrSF"]], y=train_df["SalePrice"])
# %%
opt, a, b = optimize_f_hat(
    obj_func=sm.pred_surface,
    constraint_ueq=[
        lambda x: 2000 - x[0] - x[1],
    ],
    max_iter=int(config.get("max_iter")),
    size_pop=int(config.get("size_pop")),
    x_min=x_min,
    x_max=x_max,
    opt_type=config.get("opt_type"),
)
# %%
# print(opt.best_x)
# print(opt.best_y)
# %%
plot_obj_surface(
    pso_opt=opt,
    func=sm.pred_surface,
    max_iter=int(config.get("max_iter")),
    x_max=x_max,
    x_min=x_min,
    x1_step=int(config.get("x1_step")),
    x2_step=int(config.get("x2_step")),
    animate=True,
)
# %%
