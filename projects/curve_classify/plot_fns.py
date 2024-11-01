import numpy as np
import plotly
import plotly.express
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_curve(
    curve: np.ndarray,
    label: np.ndarray,
    color_plate: dict,
    plot_file_path: str = ".",
    plot_name: str = "curve_compare_plot",
) -> None:
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
    )
    seen_label = set()

    for i, tr in enumerate(label):
        fig.add_trace(
            go.Scatter(
                x=list(range(curve.shape[2])),
                y=curve[i, 0, :],
                mode="lines",
                name="",
                line=dict(color=color_plate.get(tr)),
                legendgroup=f"label {tr}",
                legendgrouptitle={"text": f"label {tr}"},
                showlegend=tr not in seen_label,
            ),
            row=1,
            col=1,
        )
        seen_label |= set([tr])

    fig.update_layout(
        title="ECG Curve",
        title_x=0.5,
    )

    plotly.offline.plot(fig, filename=f"{plot_file_path}/{plot_name}.html")
