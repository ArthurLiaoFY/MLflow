import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_curve(
    curve: pd.DataFrame,
    label: pd.DataFrame,
    plot_file_path: str = ".",
    plot_name: str = "curve_compare_plot",
) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(
            "Negative Results",
            "Positive Results",
        ),
    )

    for i, tr in enumerate(label["test_result"]):
        fig.add_trace(
            go.Scatter(
                x=curve.columns,
                y=curve.loc[i, :],
                mode="lines",
                name="",
                line=dict(color="red" if tr == -1 else "blue"),
                legendgroup="Negative" if tr == -1 else "Positive",
                legendgrouptitle={
                    "text": "Negative" if tr == -1 else "Positive",
                },
                showlegend=False,
            ),
            row=1,
            col=1 if tr == -1 else 2,
        )

    fig.update_layout(
        title="Production Curve",
    )

    plotly.offline.plot(fig, filename=f"{plot_file_path}/{plot_name}.html")
