# %%
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ml_models.linear_models.models.adaptive_controller import (
    BetaController,
    PIDController,
)
from ml_models.linear_models.tools import to_model_matrix

target_point_1 = 43
target_1_duration = 7000
target_point_2 = 37
target_2_duration = 7000
sample_size = 50
seed = np.random.RandomState(1122)
epsilon = 0.5

upper_input_value = 45
init_input_value = 40
lower_input_value = 35

bc = BetaController(target_point=target_point_1)
pidc = PIDController(target_point=target_point_1, Kp=0.013, Ki=0.422, Kd=0.005)


def f(iv, time):
    degradation = 0.0042 * time
    return (
        1.38681004 * iv
        - 12.75095789
        + seed.uniform(low=-epsilon, high=epsilon, size=1)
        # + seed.randn() * epsilon
        - degradation
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


historical_ivs = np.concatenate(
    (
        np.repeat(a=np.array([upper_input_value]), repeats=sample_size),
        np.repeat(a=np.array([lower_input_value]), repeats=sample_size),
        np.repeat(a=np.array([init_input_value]), repeats=sample_size),
    )
).reshape((sample_size * 3, 1))
seed.shuffle(historical_ivs)
historical_ovs = np.array(
    [f(iv=iv, time=time) for time, iv in enumerate(historical_ivs)]
)


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
    model_mat = to_model_matrix(historical_ivs)
    bc.estimate_Kp(X=model_mat, y=historical_ovs)

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
    #         np.concatenate((historical_ovs, bc_control_ovs), axis=0)
    #         if bc_control_ovs is not None
    #         else historical_ovs
    #     ),
    # )

    bc_ctrl = bc.compute(
        y_new=historical_ovs[-1] if bc_control_ovs is None else bc_control_ovs[-1],
    )
    pidc_ctrl = pidc.compute(
        y_new=historical_ovs[-1] if pidc_control_ovs is None else pidc_control_ovs[-1],
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
                f(iv=non_control_ivs[-1].reshape(1, -1), time=time),
            ),
            axis=0,
        )
        if non_control_ovs is not None
        else f(iv=non_control_ivs[-1].reshape(1, -1), time=time)
    )

    bc_control_ivs = (
        np.concatenate(
            (
                bc_control_ivs,
                bc_control_ivs[-1].reshape(1, -1) + bc_ctrl,
            ),
            axis=0,
        )
        if bc_control_ivs is not None
        else historical_ivs[-1].reshape(1, -1) + bc_ctrl
    )
    bc_control_ovs = (
        np.concatenate(
            (
                bc_control_ovs,
                f(iv=bc_control_ivs[-1].reshape(1, -1), time=time),
            ),
            axis=0,
        )
        if bc_control_ovs is not None
        else f(iv=bc_control_ivs[-1].reshape(1, -1), time=time)
    )

    pidc_control_ivs = (
        np.concatenate(
            (
                pidc_control_ivs,
                pidc_control_ivs[-1].reshape(1, -1) + pidc_ctrl,
            ),
            axis=0,
        )
        if pidc_control_ivs is not None
        else historical_ivs[-1].reshape(1, -1) + pidc_ctrl
    )
    pidc_control_ovs = (
        np.concatenate(
            (
                pidc_control_ovs,
                f(iv=pidc_control_ivs[-1].reshape(1, -1), time=time),
            ),
            axis=0,
        )
        if pidc_control_ovs is not None
        else f(iv=pidc_control_ivs[-1].reshape(1, -1), time=time)
    )

    if cnt == target_1_duration - 1:
        bc.reset_target_point(
            target_point=target_point_2,
        )
        pidc = PIDController(target_point=target_point_2, Kp=0.013, Ki=0.422, Kd=0.005)

    elif cnt == target_1_duration + target_2_duration - 1:
        break
    cnt += 1
    time += 1


plot_ctrl_trend(non_control_ovs)
plot_ctrl_trend(pidc_control_ovs)
plot_ctrl_trend(bc_control_ovs)


fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=pidc_control_ivs.squeeze(),
        y=pidc_control_ovs.squeeze(),
        mode="markers",
        name="PID Controller",
        marker=dict(color="green", opacity=0.6),
    )
)
fig.add_trace(
    go.Scatter(
        x=bc_control_ivs.squeeze(),
        y=bc_control_ovs.squeeze(),
        mode="markers",
        name="Beta Controller",
        marker=dict(color="blue", opacity=0.6),
    )
)
fig.update_layout(
    title="Scatter Plot of different controller",
    xaxis_title="IVs",
    yaxis_title="OVs",
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

print("[ PID Controller ] : ")
print(
    "mean : ",
    (
        pidc_control_ovs
        - np.array(
            [target_point_1] * target_1_duration + [target_point_2] * target_2_duration
        )
    ).mean(),
)
print(
    "std : ",
    (
        pidc_control_ovs
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
