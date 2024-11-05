# %%
import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.opt_ml.estimate_surface import EstimateSurface
from projects.opt_ml.optimize_response import optimize_f_hat
from projects.opt_ml.plot_fns import plot_obj_surface

# load config
config = ConfigParser()
config.read("projects/opt_ml/opt_ml.ini")
config = config["taichung_housing"]

# load data
presale_data_list = []
secondhand_data_list = []
for path, root, files in os.walk("./data/taichung_housing"):
    for f in files:
        if f.endswith("lvr_land_A.csv"):
            secondhand_data_list.append(os.path.join(path, f))
        if f.endswith("lvr_land_B.csv"):
            presale_data_list.append(os.path.join(path, f))


presale_df = pd.concat(
    objs=(pd.read_csv(f, header=[0, 1]) for f in presale_data_list),
    axis=0,
)
presale_df(('預售中古', '_')) = '預售屋'
secondhand_df = pd.concat(
    objs=(pd.read_csv(f, header=[0, 1]) for f in secondhand_data_list),
    axis=0,
)
secondhand_df(('預售中古', '_')) = '中古屋'
# %%
presale_df_columns_map = {
    'c_'+str(idx): org_col[0]
    for idx, org_col in enumerate(presale_df.columns)
}


secondhand_df_columns_map = {
    'c_'+str(idx): org_col[0]
    for idx, org_col in enumerate(secondhand_df.columns)
}


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
