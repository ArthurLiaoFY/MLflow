# %%
from configparser import ConfigParser

from projects.opt_ml.estimate_surface import EstimateSurface
from projects.opt_ml.optimize_response import optimize_f_hat
from projects.opt_ml.plot_fns import plot_obj_surface
from projects.opt_ml.simulate_data import SimulateData

config = ConfigParser()
config.read("projects/opt_ml/opt_ml.ini")
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
sm = EstimateSurface(run_id="simulate", **config)
sm.fit_surface(X=X_train, y=y_train)
# %%
y_hat = sm.pred_surface(valid_X=X_val)


opt, a, b = optimize_f_hat(
    obj_func=sm.pred_surface,
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
    func=sm.pred_surface,
    max_iter=int(config.get("max_iter")),
    x_max=sd.x_max,
    x_min=sd.x_min,
    x1_step=int(config.get("x1_step")),
    x2_step=int(config.get("x2_step")),
    animate=True,
    desc=config.get("func"),
)

# %%
