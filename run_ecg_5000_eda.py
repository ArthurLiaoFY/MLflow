# %%
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import arff

# %%

config = ConfigParser()
config.read("projects/ecg_5000/ecg_5000.ini")
# %%
df = pd.DataFrame(arff.loadarff("./data/ECG5000/ECG5000_TRAIN.arff")[0])
# %%
for t in [b"1", b"2", b"3", b"4", b"5"]:
    for i in range(2):
        plt.plot(df.loc[df["target"] == t, :].iloc[i, :], "o-")
    plt.show()

# %%


def plot_curve(
    curve: pd.DataFrame,
    label: pd.DataFrame,
    plot_file_path: str = ".",
) -> None:
    fig = make_subplots(
        rows=1,
        cols=5,
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

    plotly.offline.plot(fig, filename=f"{plot_file_path}/curve_compare_plot.html")
