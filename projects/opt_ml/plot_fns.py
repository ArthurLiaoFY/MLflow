import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D


def plot_obj_surface(
    pso_opt,
    func,
    max_iter: int,
    x_max: list,
    x_min: list,
    x1_step: float,
    x2_step: float,
    animate: bool = False,
    desc: str = "housing",
):
    x1_grid = np.linspace(
        start=x_min[0],
        stop=x_max[0],
        num=x1_step,
    )
    x2_grid = np.linspace(
        start=x_min[1],
        stop=x_max[1],
        num=x2_step,
    )
    X, Y = np.meshgrid(
        x1_grid,
        x2_grid,
    )
    # Z = f(X, Y)
    Z_hat = func(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
        len(x1_grid), len(x2_grid)
    )

    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2)

    ax0.set_aspect("auto")
    ax0.plot_surface(
        X,
        Y,
        Z_hat,
        cmap=plt.cm.coolwarm,
    )
    # scatter_3d = Line3D(
    #     [],
    #     [],
    #     [],
    #     marker="o",
    #     color="black",
    #     markersize=5,
    #     linestyle="None",
    # )
    # ax0.add_line(scatter_3d)

    ax0.set_xlabel(r"x1")
    ax0.set_ylabel(r"x2")
    ax0.set_title(
        f"Simulated Surface",
        fontweight="bold",
        fontsize=15,
    )
    ax0.view_init(elev=20.0, azim=-45)

    ax1.contour(X, Y, Z_hat, cmap=plt.cm.coolwarm)
    scatter_2d = ax1.plot(
        [],
        [],
        marker="o",
        color="black",
        markersize=5,
        linestyle="None",
    )
    match desc:
        case "simulate":
            pass

        case "housing":
            # go housing
            ax1.axline(
                xy1=(20, 180),
                xy2=(40, 160),
                color="r",
                linestyle="--",
                lw=1,
            )
    ax1.set_title(
        "Contour Plot",
        fontweight="bold",
        fontsize=15,
    )
    ax1.set_xlabel(r"x1")
    ax1.set_ylabel(r"x2")

    if animate:

        # def update_surface(frame):
        #     i, j = frame // 10, frame % 10
        #     ax1.set_title("iter = " + str(i))
        #     X_tmp = (
        #         pso_opt.record_value["X"][i] + pso_opt.record_value["V"][i] * j / 10.0
        #     )
        #     scatter_3d.set_data(X_tmp[:, 0], X_tmp[:, 1])
        #     scatter_3d.set_3d_properties(f_hat.predict(X_tmp))

        #     return (scatter_3d,)

        def update_contour(frame):
            i, j = frame // 10, frame % 10
            ax1.set_title(
                "iter = " + str(i),
                fontweight="bold",
                fontsize=15,
            )
            X_tmp = (
                pso_opt.record_value["X"][i] + pso_opt.record_value["V"][i] * j / 10.0
            )
            plt.setp(scatter_2d, "xdata", X_tmp[:, 0], "ydata", X_tmp[:, 1])

            return (scatter_2d,)

        def update(frame):
            # update_surface(frame=frame)
            update_contour(frame=frame)

        ani = FuncAnimation(
            fig=fig,
            func=update,
            blit=False,
            interval=25,
            frames=max_iter * 10,
        )

    plt.tight_layout()
    plt.show()

    # ani.save("pso.gif", writer="pillow")
