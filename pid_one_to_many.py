# %%
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ml_models.linear_models.models.adaptive_controller import BetaController
from ml_models.linear_models.tools import to_model_matrix

target_point_1 = -30
target_1_duration = 700
target_point_2 = -40
target_2_duration = 700
sample_size = 100
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
    degradation = 0.0042 * time

    return (
        50
        + 1.5 * iv[0]
        - 3.2 * iv[1]
        + seed.uniform(low=-epsilon, high=epsilon, size=1)
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
        np.array([[35, 45]] * sample_size),
        np.array([[35, 35]] * sample_size),
        np.array([[35, 25]] * sample_size),
        np.array([[25, 45]] * sample_size),
        np.array([[25, 35]] * sample_size),
        np.array([[25, 25]] * sample_size),
        np.array([[15, 45]] * sample_size),
        np.array([[15, 35]] * sample_size),
        np.array([[15, 25]] * sample_size),
    )
).reshape((sample_size * 9, 2))
seed.shuffle(historical_ivs)
historical_ovs = np.array(
    [f(iv=iv, time=time) for time, iv in enumerate(historical_ivs)]
)
# %%
plt.plot(historical_ivs[:, 0], historical_ivs[:, 1], "o")
plt.show()
plt.plot(historical_ivs[:, 0], historical_ovs, "o")
plt.show()
plt.plot(historical_ivs[:, 1], historical_ovs, "o")
plt.show()
# %%
X = to_model_matrix(historical_ivs)
print(np.linalg.pinv(X.T @ X) @ X.T @ historical_ovs)

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
