# %%
import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.opt_ml.estimate_surface import EstimateSurface
from projects.opt_ml.load_data import load_presale_data
from projects.opt_ml.optimize_response import optimize_f_hat
from projects.opt_ml.plot_fns import plot_obj_surface
from projects.opt_ml.trans_data import TransformData

# load config
config = ConfigParser()
config.read("projects/opt_ml/opt_ml.ini")
config = config["taiwan_housing"]

trans_cls = TransformData(
    dismiss_amount=100,
)

presale_df = load_presale_data(data_file_path=config.get("data_file_path"))

# %%
trans_df = trans_cls.fit_transform(df=presale_df)
# %%
b = trans_cls.fit_transform(df=presale_df.iloc[[0], :])
# %%
sm = EstimateSurface(
    run_id="boston_housing",
    in_feature=trans_df.shape[-1] - 1,
    **config,
)
sm.fit_surface(
    X=trans_df[:, :-1],
    y=trans_df[:, -1],
)
# %%
price_pred = sm.pred_surface(
    valid_X=trans_df[:, :-1],
)
# %%
plt.hist(
    trans_df[:, -1],
    bins=np.arange(start=0.0, stop=trans_df[:, -1].max() + 5e5, step=5e5),
)
plt.hist(
    price_pred, bins=np.arange(start=0.0, stop=trans_df[:, -1].max() + 5e5, step=5e5)
)
plt.show()
# %%
plt.scatter(x=price_pred, y=trans_df[:, -1], c="k")
plt.show()
# %%
plt.plot(
    trans_df[:, -1],
    trans_df[:, -1] - price_pred,
    "o",
    c="k",
)
plt.show()
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
    proj_case="housing",
)
# %%
# print(opt.best_x)
# print(opt.best_y)
# %%
# plot_obj_surface(
#     pso_opt=opt,
#     func=sm.pred_surface,
#     max_iter=int(config.get("max_iter")),
#     x_max=x_max,
#     x_min=x_min,
#     x1_step=int(config.get("x1_step")),
#     x2_step=int(config.get("x2_step")),
#     animate=True,
# )
# %%
