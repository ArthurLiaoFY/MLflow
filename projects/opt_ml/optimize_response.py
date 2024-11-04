# %%
import matplotlib.pyplot as plt
import numpy as np
from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO


# %%
def optimize_f_hat(
    obj_func,
    constraint_ueq: list,
    max_iter: int,
    size_pop: int,
    x_min: list,
    x_max: list,
    opt_type: str = "DE",
):


    def func(x: np.ndarray):
        return float(obj_func(np.array([x])).item())

    match opt_type:
        case "DE":
            opt = DE(
                func=func,
                n_dim=len(x_max),
                size_pop=size_pop,
                max_iter=max_iter,
                lb=x_min,
                ub=x_max,
                constraint_ueq=constraint_ueq,
            )

            best_x, best_obj_func = opt.run()
            return opt, best_x, best_obj_func.item()

        case "GA":
            opt = GA(
                func=func,
                n_dim=len(x_max),
                size_pop=size_pop,
                max_iter=max_iter,
                lb=x_min,
                ub=x_max,
                constraint_ueq=constraint_ueq,
            )
            best_x, best_obj_func = opt.run()
            return opt, best_x, best_obj_func.item()

        case "PSO":
            opt = PSO(
                func=func,
                n_dim=len(x_max),
                pop=size_pop,
                max_iter=max_iter,
                lb=x_min,
                ub=x_max,
                constraint_ueq=constraint_ueq,
            )
            opt.record_mode = True
            opt.run()
            return opt, opt.gbest_x, opt.gbest_y.item()

        case _:
            raise NotImplementedError
