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

# load config
config = ConfigParser()
config.read("projects/opt_ml/opt_ml.ini")
config = config["taiwan_housing"]
presale_df = load_presale_data(data_file_path=config.get("data_file_path"))

# %%
trans_df = pd.concat(
    objs=(
        pd.get_dummies(data=presale_df["城市"], prefix="城市"),
        pd.get_dummies(data=presale_df["鄉鎮市區"].fillna("其他")),
        pd.DataFrame.from_dict(
            {
                idx: dict(zip(key_l, value_l))
                for idx, key_l, value_l in zip(
                    presale_df.index,
                    presale_df.loc[:, "交易筆棟數"].str.findall(r"[^\d]+").to_numpy(),
                    presale_df.loc[:, "交易筆棟數"].str.findall(r"\d+").to_numpy(),
                )
            },
            orient="index",
        ).add_suffix("數量"),
        pd.get_dummies(presale_df.get("建物型態").fillna("其他"), prefix="建物型態_"),
        pd.get_dummies(presale_df.get("記錄年份"), prefix="記錄年份"),
        pd.get_dummies(presale_df.get("記錄季度"), prefix="記錄季度"),
        presale_df.get("土地移轉總面積平方公尺"),
        presale_df.get("建物移轉總面積平方公尺"),
        presale_df.get("車位移轉總面積平方公尺"),
        presale_df.get("建物現況格局-房"),
        presale_df.get("建物現況格局-廳"),
        presale_df.get("建物現況格局-衛"),
        presale_df.get("建物現況格局-隔間").apply(lambda x: 1 if x == "有" else 0),
        ##############
        presale_df.get("總價元"),
    ),
    axis=1,
).astype(float)

# %%
sm = EstimateSurface(run_id="boston_housing", **config)
sm.fit_surface(X=trans_df.drop(columns=["總價元"]), y=trans_df["總價元"])
price_pred = sm.pred_surface(valid_X=trans_df.drop(columns=["總價元"]))
# %%
plt.hist(
    trans_df["總價元"],
    bins=np.arange(start=0.0, stop=trans_df["總價元"].max() + 5e5, step=5e5),
)
plt.hist(
    price_pred, bins=np.arange(start=0.0, stop=trans_df["總價元"].max() + 5e5, step=5e5)
)
plt.show()
# %%
plt.scatter(x=price_pred, y=trans_df["總價元"], c="k")
plt.show()
# %%
plt.plot(
    trans_df["總價元"],
    trans_df["總價元"] - price_pred,
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
