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
    ignore_index=True,
)
presale_df[("預售中古", "_")] = 1

secondhand_df = pd.concat(
    objs=(pd.read_csv(f, header=[0, 1]) for f in secondhand_data_list),
    axis=0,
    ignore_index=True,
)
secondhand_df[("預售中古", "_")] = 0

raw_df = pd.concat(
    objs=(presale_df, secondhand_df),
    axis=0,
    ignore_index=True,
).set_index(("編號", "serial number"))
raw_df.columns = raw_df.columns.get_level_values(0)
# %%
tmp = pd.concat(
    objs=(
        pd.get_dummies(data=raw_df["鄉鎮市區"].fillna("其他")).add_prefix("台中市"),
        pd.DataFrame.from_dict(
            {
                idx: dict(zip(key_l, value_l))
                for idx, key_l, value_l in zip(
                    raw_df.index,
                    raw_df.loc[:, "交易筆棟數"].str.findall(r"[^\d]+").to_numpy(),
                    raw_df.loc[:, "交易筆棟數"].str.findall(r"\d+").to_numpy(),
                )
            },
            orient="index",
        ).add_suffix("數量"),
        pd.get_dummies(
            raw_df["都市土地使用分區"].apply(
                lambda x: (
                    "土地使用分區_其他"
                    if x not in ("住", "商", "工", "農")
                    else "土地使用分區_" + x
                )
            )
        ),
        raw_df.get("土地移轉總面積平方公尺"),
        pd.get_dummies(raw_df.get("建物型態").fillna("其他")).add_prefix("建物型態_"),
        raw_df.get("建物移轉總面積平方公尺"),
        raw_df.get("建物現況格局-房"),
        raw_df.get("建物現況格局-廳"),
        raw_df.get("建物現況格局-衛"),
        raw_df.get("建物現況格局-隔間").apply(lambda x: 1 if x == "有" else 0),
        raw_df.get("有無管理組織").apply(lambda x: 1 if x == "有" else 0),
        raw_df.get("車位類別").apply(lambda x: 1 if x == "有" else 0),
        ##############
        raw_df.get("解約情形").fillna("無").apply(lambda x: 1 if "解約" in x else 0),
        raw_df.get("預售中古"),
        raw_df.get("主建物面積").fillna(0.0),
        raw_df.get("附屬建物面積").fillna(0.0),
        raw_df.get("陽台面積").fillna(0.0),
        raw_df.get("電梯").fillna("無").apply(lambda x: 1 if x == "有" else 0),
        ##############
        raw_df.get("總價元"),
    ),
    axis=1,
)

# %%
tmp.columns

# %%

# %%
[c[0] for c in raw_df.columns]
# %%
# np.unique(
#     total_df[("都市土地使用分區", "the use zoning or compiles and checks")].astype(
#         "str"
#     ),
#     return_counts=True,
# )
# %%
df_columns_map = {
    "c_" + str(idx): org_col[0] for idx, org_col in enumerate(raw_df.columns)
}
raw_df.columns = ["c_" + str(idx) for idx in range(len(raw_df.columns))]

# %%

# %%

# %%

# sm = EstimateSurface(run_id="boston_housing", **config)
# sm.fit_surface(X=train_df[["1stFlrSF", "2ndFlrSF"]], y=train_df["SalePrice"])
# %%
# opt, a, b = optimize_f_hat(
#     obj_func=sm.pred_surface,
#     constraint_ueq=[
#         lambda x: 2000 - x[0] - x[1],
#     ],
#     max_iter=int(config.get("max_iter")),
#     size_pop=int(config.get("size_pop")),
#     x_min=x_min,
#     x_max=x_max,
#     opt_type=config.get("opt_type"),
# )
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
