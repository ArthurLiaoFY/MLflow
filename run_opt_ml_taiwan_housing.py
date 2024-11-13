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
offset = {
    "城市": "臺中市",
    "鄉鎮市區": "西區",
    "土地數量": "1",
    "建物數量": "1",
    "車位數量": "1",
    "建物型態": "住宅大樓(11層含以上有電梯)",
    "記錄季度": "S3",
    "土地移轉總面積平方公尺": "23.0",
    "建物現況格局-房": "3",
    "建物現況格局-廳": "2",
    "建物現況格局-衛": "2",
    "建物現況格局-隔間": "1",
}


def obj_func(x: np.ndarray):
    global offset, trans_cls
    return sm.pred_surface(
        trans_cls.fit_transform(
            df=pd.DataFrame.from_dict(
                {
                    idx: {
                        **offset,
                        "車位移轉總面積平方公尺": sample[0],
                        "建物移轉總面積平方公尺": sample[1],
                    }
                    for idx, sample in enumerate(x.reshape(1, -1) if x.ndim == 1 else x)
                },
                orient="index",
            ),
            inference=True,
        )
    )


# %%
opt, a, b = optimize_f_hat(
    obj_func=obj_func,
    constraint_ueq=[
        lambda x: 200 - x[0] - x[1],
        lambda x: 20 - x[0],
        lambda x: 10 - x[1],
    ],
    max_iter=int(config.get("max_iter")),
    size_pop=int(config.get("size_pop")),
    x_min=[20, 10],
    x_max=[40, 200],
    opt_type=config.get("opt_type"),
)

# %%
plot_obj_surface(
    pso_opt=opt,
    func=obj_func,
    max_iter=int(config.get("max_iter")),
    x_min=[20, 10],
    x_max=[40, 200],
    x1_step=int(config.get("x1_step")),
    x2_step=int(config.get("x2_step")),
    animate=True,
)
# %%
