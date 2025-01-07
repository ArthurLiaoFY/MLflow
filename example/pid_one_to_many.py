# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from itertools import product

import numpy as np
import plotly.graph_objects as go

from ml_models.linear_models.models.adaptive_controller import BetaController
from ml_models.linear_models.tools import to_model_matrix

target_point_1 = -30
target_1_duration = 1500
target_point_2 = -40
target_2_duration = 1500
sample_size = 5
seed = np.random.RandomState(1122)
epsilon = 0.5
fit_model_sample_size = 5000

upper_iv = [35, 45]
init_iv = [25, 35]
lower_iv = [15, 25]

learning_rate = 1
bc = BetaController(target_point=target_point_1)


# assume linear relationship
def f(iv, time):
    degradation = 0.01 * time

    return np.array(
        [
            50
            + 1.5 * iv[0]
            - 3.2 * iv[1]
            + seed.uniform(low=-epsilon, high=epsilon, size=1)
            # + seed.randn() * epsilon
            - degradation
        ]
    )


def plot_ctrl_trend(control_ovs):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(control_ovs.squeeze()))),
            y=control_ovs.squeeze(),
            mode="lines+markers",
            name="Control OVs",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration)),
            y=[target_point_1] * target_1_duration,
            mode="lines",
            name="Target Point 1",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration, target_1_duration + target_2_duration)),
            y=[target_point_2] * target_2_duration,
            mode="lines",
            name="Target Point 2",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration)) + list(range(target_1_duration))[::-1],
            y=(
                list(target_point_1 - epsilon for _ in range(target_1_duration))
                + list(target_point_1 + epsilon for _ in range(target_1_duration))[::-1]
            ),
            fill="toself",
            fillcolor="rgba(0, 255, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Acceptable Range",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration, target_1_duration + target_2_duration))
            + list(range(target_1_duration, target_1_duration + target_2_duration))[
                ::-1
            ],
            y=(
                list(
                    target_point_2 - epsilon
                    for _ in range(
                        target_1_duration, target_1_duration + target_2_duration
                    )
                )
                + list(
                    target_point_2 + epsilon
                    for _ in range(
                        target_1_duration, target_1_duration + target_2_duration
                    )
                )[::-1]
            ),
            fill="toself",
            fillcolor="rgba(0, 255, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Acceptable Range",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration)) + list(range(target_1_duration))[::-1],
            y=(
                list(target_point_1 - epsilon for _ in range(target_1_duration))
                + list(target_point_1 - 3 * epsilon for _ in range(target_1_duration))[
                    ::-1
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Lower Danger Zone",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration, target_1_duration + target_2_duration))
            + list(range(target_1_duration, target_1_duration + target_2_duration))[
                ::-1
            ],
            y=(
                list(
                    target_point_2 - epsilon
                    for _ in range(
                        target_1_duration, target_1_duration + target_2_duration
                    )
                )
                + list(
                    target_point_2 - 3 * epsilon
                    for _ in range(
                        target_1_duration, target_1_duration + target_2_duration
                    )
                )[::-1]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Lower Danger Zone",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration)) + list(range(target_1_duration))[::-1],
            y=(
                list(target_point_1 + epsilon for _ in range(target_1_duration))
                + list(target_point_1 + 3 * epsilon for _ in range(target_1_duration))[
                    ::-1
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Upper Danger Zone",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(target_1_duration, target_1_duration + target_2_duration))
            + list(range(target_1_duration, target_1_duration + target_2_duration))[
                ::-1
            ],
            y=(
                list(
                    target_point_2 + epsilon
                    for _ in range(
                        target_1_duration, target_1_duration + target_2_duration
                    )
                )
                + list(
                    target_point_2 + 3 * epsilon
                    for _ in range(
                        target_1_duration, target_1_duration + target_2_duration
                    )
                )[::-1]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Upper Danger Zone",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Control OVs Plot",
        xaxis_title="Index",
        yaxis_title="Values",
        template="plotly_white",
    )

    fig.show()


historical_ivs = (
    np.array(
        list(
            product(
                *(
                    np.repeat(np.linspace(lower, upper, 5), sample_size)
                    for lower, upper in zip(lower_iv, upper_iv)
                )
            )
        )
    )
    .repeat(sample_size, axis=0)
    .reshape(-1, 2)
)

seed.shuffle(historical_ivs)
historical_ovs = np.array(
    [f(iv=iv, time=time).item() for time, iv in enumerate(historical_ivs)]
)

# %%
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=historical_ivs[:, 0],
        y=historical_ivs[:, 1],
        mode="markers",
        name="Experiment Points",
        marker=dict(color="blue", size=5, opacity=0.7),
    )
)
fig.update_layout(
    title="Experiment Points",
    xaxis_title="IV 1",
    yaxis_title="IV 2",
    template="plotly_white",
)

fig.show()

# %%
non_control_ivs = None
non_control_ovs = None

bc_control_ivs = None
bc_control_ovs = None

pidc_control_ivs = None
pidc_control_ovs = None
time = len(historical_ivs)
# %%

cnt = 0
while True:
    bc.estimate_parameter(X=historical_ivs, y=historical_ovs)
    # model_mat = to_model_matrix(
    #     X=(
    #         np.concatenate((historical_ivs, bc_control_ivs), axis=0)
    #         if bc_control_ivs is not None
    #         else historical_ivs
    #     )
    # )
    # bc.estimate_Kp(
    #     X=model_mat,
    #     y=(
    #         np.concatenate((historical_ovs, bc_control_ovs[:, np.newaxis]), axis=0)
    #         if bc_control_ovs is not None
    #         else historical_ovs
    #     ),
    # )
    bc_ctrl = bc.compute(
        y_new=historical_ovs[-1] if bc_control_ovs is None else bc_control_ovs[-1],
    )

    non_control_ivs = (
        np.concatenate(
            (
                non_control_ivs,
                non_control_ivs[-1].reshape(1, -1),
            ),
            axis=0,
        )
        if non_control_ivs is not None
        else historical_ivs[-1].reshape(1, -1)
    )
    non_control_ovs = (
        np.concatenate(
            (
                non_control_ovs,
                f(iv=non_control_ivs[-1], time=time),
            ),
            axis=0,
        )
        if non_control_ovs is not None
        else f(iv=non_control_ivs[-1], time=time)
    )

    bc_control_ivs = (
        np.concatenate(
            (
                bc_control_ivs,
                bc_control_ivs[-1].reshape(1, -1) + bc_ctrl.T,
            ),
            axis=0,
        )
        if bc_control_ivs is not None
        else historical_ivs[-1].reshape(1, -1) + bc_ctrl.T
    )
    bc_control_ovs = (
        np.concatenate(
            (
                bc_control_ovs,
                f(iv=bc_control_ivs[-1], time=time),
            ),
            axis=0,
        )
        if bc_control_ovs is not None
        else f(iv=bc_control_ivs[-1], time=time)
    )

    if cnt == target_1_duration - 1:
        bc.reset_target_point(
            target_point=target_point_2,
        )

    elif cnt == target_1_duration + target_2_duration - 1:
        break
    cnt += 1
    time += 1


plot_ctrl_trend(non_control_ovs)
plot_ctrl_trend(bc_control_ovs)
# %%
fig = go.Figure()

x_min, x_max = 15, 35
y_min, y_max = 25, 45
z_min, z_max = np.min(bc_control_ovs), np.max(bc_control_ovs)


vertices = [
    [x_min, y_min, z_min],
    [x_max, y_min, z_min],
    [x_max, y_max, z_min],
    [x_min, y_max, z_min],
    [x_min, y_min, z_max],
    [x_max, y_min, z_max],
    [x_max, y_max, z_max],
    [x_min, y_max, z_max],
]

i, j, k = zip(
    (0, 1, 2),
    (0, 2, 3),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)
inside_rect = (
    (bc_control_ivs[:, 0] >= x_min)
    & (bc_control_ivs[:, 0] <= x_max)
    & (bc_control_ivs[:, 1] >= y_min)
    & (bc_control_ivs[:, 1] <= y_max)
)

colors = np.where(inside_rect, "blue", "red")

fig.add_trace(
    go.Scatter3d(
        x=bc_control_ivs[:, 0],
        y=bc_control_ivs[:, 1],
        z=bc_control_ovs.squeeze(),
        mode="markers",
        marker=dict(size=1, color=colors, opacity=0.8),
        name="Data Points",
    )
)

fig.add_trace(
    go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=[v[2] for v in vertices],
        i=i,
        j=j,
        k=k,
        opacity=0.1,
        color="green",
        name="Rectangle",
    )
)

fig.update_layout(
    title="3D Scatter Plot of Beta Controller",
    scene=dict(
        xaxis_title="IV 1",
        yaxis_title="IV 2",
        zaxis_title="OV",
    ),
    template="plotly_white",
)

fig.show()


# %%
print("[ Non Controller ] : ")
print(
    "mean : ",
    (
        non_control_ovs
        - np.array(
            [target_point_1] * target_1_duration + [target_point_2] * target_2_duration
        )
    ).mean(),
)
print(
    "std : ",
    (
        non_control_ovs
        - np.array(
            [target_point_1] * target_1_duration + [target_point_2] * target_2_duration
        )
    ).std(),
)


print("[ Beta Controller ] : ")
print(
    "mean : ",
    (
        bc_control_ovs
        - np.array(
            [target_point_1] * target_1_duration + [target_point_2] * target_2_duration
        )
    ).mean(),
)
print(
    "std : ",
    (
        bc_control_ovs
        - np.array(
            [target_point_1] * target_1_duration + [target_point_2] * target_2_duration
        )
    ).std(),
)
# %%
