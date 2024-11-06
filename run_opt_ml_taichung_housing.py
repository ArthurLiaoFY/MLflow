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
presale_year_list = []

secondhand_data_list = []
secondhand_year_list = []

for path, root, files in os.walk("./data/taichung_housing"):
    for f in files:
        if f.endswith("lvr_land_A.csv"):
            secondhand_year_list.append(path.split("/")[-1][:3])
            secondhand_data_list.append(os.path.join(path, f))

        if f.endswith("lvr_land_B.csv"):
            presale_year_list.append(path.split("/")[-1][:3])
            presale_data_list.append(os.path.join(path, f))

# %%
presale_df = (
    pd.concat(
        objs=(pd.read_csv(f, header=[0]).iloc[1:, :] for f in presale_data_list),
        axis=0,
        keys=presale_year_list,
    )
    .reset_index(level=[0])
    .rename(columns={"level_0": "記錄年份"})
)
presale_df["建築完成年月"] = np.nan
presale_df["預售中古"] = 1

secondhand_df = (
    pd.concat(
        objs=(pd.read_csv(f, header=[0]).iloc[1:, :] for f in secondhand_data_list),
        axis=0,
        keys=secondhand_year_list,
    )
    .reset_index(level=[0])
    .rename(columns={"level_0": "記錄年份"})
)
secondhand_df["預售中古"] = 0
secondhand_df["建築完成年月"] = (
    secondhand_df.get("建築完成年月").astype(str).apply(lambda x: x[:3])
)
raw_df = pd.concat(
    objs=(presale_df, secondhand_df),
    axis=0,
    ignore_index=True,
).set_index("編號")

# %%
trans_df = pd.concat(
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
        ##############
        raw_df.get("建築完成年月")
        .isna()
        .astype(int)
        .rename({"建築完成年月": "是否有記錄建築完成年月"}),
        pd.get_dummies(
            data=(
                raw_df.get("記錄年份").astype(float)
                - raw_df.get("建築完成年月").astype(float)
            ).apply(
                lambda x: (
                    "少於五年"
                    if x <= 5
                    else (
                        "少於十年"
                        if x <= 10
                        else (
                            "少於十五年"
                            if x <= 15
                            else (
                                "少於二十年"
                                if x <= 20
                                else (
                                    "少於二十五年"
                                    if x <= 25
                                    else "少於三十年" if x <= 30 else "其他"
                                )
                            )
                        )
                    )
                )
            ),
            prefix="屋齡",
        ),
        ##############
        raw_df.get("土地移轉總面積平方公尺"),
        raw_df.get("建物移轉總面積平方公尺"),
        raw_df.get("車位移轉總面積平方公尺"),
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
).astype(float)


# %%
len(trans_df.columns)

# %%

# %%

sm = EstimateSurface(run_id="boston_housing", **config)
sm.fit_surface(X=trans_df.drop(columns=["總價元"]), y=trans_df["總價元"])
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
