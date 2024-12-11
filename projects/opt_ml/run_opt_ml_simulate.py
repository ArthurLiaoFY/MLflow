# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configparser import ConfigParser

import numpy as np
from estimate_surface import EstimateSurface
from optimize_response import optimize_f_hat
from plot_fns import plot_obj_surface
from simulate_data import SimulateData

config = ConfigParser()
config.read("./opt_ml.ini")
config = config["simulate"]


sd = SimulateData(
    seed=int(config.get("seed")),
    simulate_size=int(config.get("simulate_size")),
    variation_ratio=float(config.get("variation_ratio")),
    sim_func=config.get("func"),
)
X_train, X_val, y_train, y_val = sd.get_data(
    train_size_ratio=float(config.get("train_size_ratio"))
)
# %%
sm = EstimateSurface(run_id="simulate", in_feature=2, **config)
sm.fit_surface(X=X_train, y=y_train)
# %%
y_hat = sm.pred_surface(valid_X=X_val)


def obj_func(x: np.ndarray):
    pred = sm.pred_surface(x.reshape(1, -1) if x.ndim == 1 else x)
    return pred.item() if len(pred) == 1 else pred


opt, a, b = optimize_f_hat(
    obj_func=obj_func,
    constraint_ueq=sd.constraint_ueq,
    max_iter=int(config.get("max_iter")),
    size_pop=int(config.get("size_pop")),
    x_max=sd.x_max,
    x_min=sd.x_min,
    opt_type=config.get("opt_type"),
)
# %%
plot_obj_surface(
    pso_opt=opt,
    func=obj_func,
    max_iter=int(config.get("max_iter")),
    x_max=sd.x_max,
    x_min=sd.x_min,
    x1_step=int(config.get("x1_step")),
    x2_step=int(config.get("x2_step")),
    plot_file_path=config.get("plot_file_path"),
    animate=True,
    desc="simulate_" + config.get("func"),
)

# %%
