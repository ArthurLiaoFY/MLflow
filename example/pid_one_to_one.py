# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import plotly.graph_objects as go

from ml_models.common.tools import moving_average
from ml_models.linear_models.models.adaptive_controller import (
    BetaController,
    PIDController,
)

target_point_1 = 43
target_1_duration = 7000
target_point_2 = 37
target_2_duration = 7000
sample_size = 100
seed = np.random.RandomState(1122)
epsilon = 0.5
repair_gap_per_items = 3000
upper_input_value = 70
init_input_value = 40
lower_input_value = 35

bc = BetaController(target_point=target_point_1)
pidc = PIDController(target_point=target_point_1, Kp=0.013, Ki=0.422, Kd=0.005)


def f(iv, time):
    center_drift = 0.002 * (time % repair_gap_per_items)
    trend_drift = 0.00001 * time
    return 1.38 * (1 + trend_drift) * iv - 12.75 - center_drift + seed.randn() * epsilon


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


estimated_intercept = []
estimated_trend = []
estimated_sigma_hat = []

non_control_ivs = None
non_control_ovs = None

bc_control_ivs = None
bc_control_ovs = None

bc_influential_points_mask = []
online_prediction_error = []
retrain_traj = [False]

pidc_control_ivs = None
pidc_control_ovs = None
time = len(historical_ivs)
# %%
cnt = 0
while True:
    if cnt == 0:
        bc.estimate_Kp(X=historical_ivs, y=historical_ovs)
    elif retrain_traj[-1]:
        bc.estimate_Kp(
            X=(
                np.concatenate(
                    (historical_ivs, bc_control_ivs[bc_influential_points_mask][-300:]),
                    axis=0,
                )
            ),
            y=(
                np.concatenate(
                    (historical_ovs, bc_control_ovs[bc_influential_points_mask][-300:]),
                    axis=0,
                )
            ),
        )

    estimated_intercept.append(bc.intercept)
    estimated_trend.append(bc.trend.item())
    estimated_sigma_hat.append(bc.lm.sigma_hat)

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

    online_prediction_error.append(
        np.power(bc.lm.predict(X=bc_control_ivs[-1]) - bc_control_ovs[-1], 2).mean()
    )

    if online_prediction_error[-1] >= bc.lm.sigma_hat * 2:
        bc_influential_points_mask.append(False)
    else:
        bc_influential_points_mask.append(True)

    if np.mean(online_prediction_error[-10:]) >= bc.lm.sigma_hat:
        retrain_traj.append(True)
    else:
        retrain_traj.append(False)

    if cnt == target_1_duration - 1:
        pidc.reset_target_point(
            target_point=target_point_2,
        )
        bc.reset_target_point(
            target_point=target_point_2,
        )

    elif cnt == target_1_duration + target_2_duration - 1:
        break

    cnt += 1
    time += 1

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=estimated_intercept,
        mode="lines+markers",
        name="Estimated Intercept Trend",
        marker=dict(color="green", opacity=0.6),
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=-(
            12.75
            + 0.002
            * (np.array(range(len(historical_ivs), time)) % repair_gap_per_items)
        ),
        mode="lines",
        name="Actual Intercept Trend",
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=estimated_trend,
        mode="lines+markers",
        name="Estimated Beta Trend",
        marker=dict(color="blue", opacity=0.6),
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=1.38 * (1 + 0.00001 * np.array(range(len(historical_ivs), time))),
        mode="lines",
        name="Actual Beta Trend",
    )
)
fig.show()
# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        y=retrain_traj[:1000],
        mode="lines+markers",
    )
)
# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=moving_average(y=online_prediction_error[:1000], window_size=10),
        mode="lines+markers",
        name="Smoothed (MA30) Online Prediction Error",
        marker=dict(color="blue", opacity=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=online_prediction_error[:1000],
        mode="markers",
        name="Online Prediction Error",
        marker=dict(color="green", opacity=0.3),
    )
)

fig.add_trace(
    go.Scatter(
        x=list(range(len(historical_ivs), time)),
        y=estimated_sigma_hat[:1000],
        mode="markers",
        name="Estimated Sigma",
        marker=dict(color="red", opacity=0.3),
    )
)

fig.show()

# %%

plot_ctrl_trend(non_control_ovs)
# plot_ctrl_trend(pidc_control_ovs)
plot_ctrl_trend(bc_control_ovs)

# %%
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

fig.add_vline(
    x=lower_input_value,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Upper Input Value = {lower_input_value}",
    annotation_position="top left",
)

fig.add_vline(
    x=upper_input_value,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Upper Input Value = {upper_input_value}",
    annotation_position="top left",
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

# %%
