import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_wafer(
    df: pd.DataFrame,
    wafer_id: str,
    plot_file_path: str = ".",
):
    fig = plt.figure(figsize=(9, 7))
    plt.scatter(
        df.loc[df["wafer_id"] == wafer_id, "posx"],
        df.loc[df["wafer_id"] == wafer_id, "posy"],
        s=5,
        c=df.loc[df["wafer_id"] == wafer_id, "OVL_Y"].apply(
            lambda x: -3 if x <= -3 else 3 if x >= 3 else x
        ),
        cmap=plt.cm.coolwarm,
    )
    plt.plot(
        150 * np.cos(np.linspace(0, 2 * np.pi, 100)),
        150 * np.sin(np.linspace(0, 2 * np.pi, 100)),
        color="black",
        linestyle="-",
        lw=1,
    )
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")

    plt.colorbar(label="overlay")
    plt.savefig(f"{plot_file_path}/{wafer_id}_overlay.png")


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_factor: list[str],
    plot_file_path: str = ".",
):
    color_map = px.colors.qualitative.Set1
    fig = go.Figure()
    for idx, (key, sub_df) in enumerate(df.groupby(group_factor)):
        fig.add_trace(
            go.Scatter(
                x=sub_df[x],
                y=sub_df[y],
                mode="markers",
                marker=dict(color=color_map[idx % len(color_map)]),  # 分配颜色
                name=str(key),
            )
        )

    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title=", ".join(group_factor),
        template="plotly_white",
    )

    plotly.offline.plot(
        fig,
        filename=f"{plot_file_path}/{x}_{y}_under_{', '.join(group_factor)}_scatterplot.html",
    )


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    group_factor: list[str],
    plot_file_path: str = ".",
):
    fig = make_subplots(
        rows=1,
        cols=df[group_factor[0]].nunique(),
        shared_yaxes=True,
        subplot_titles=[
            f"{group_factor[0]}: {group}" for group in df[group_factor[0]].unique()
        ],
    )
    for idx, (eqp, sub_df) in enumerate(df.groupby(group_factor[0])):

        fig.add_trace(
            go.Box(
                x=sub_df[group_factor[1]],
                y=sub_df[x],
                name=eqp,
                boxmean=True,
            ),
            row=1,
            col=idx + 1,
        )
    fig.update_layout(
        title_text=f"Boxplot of {x} by {', '.join(group_factor)}",
        template="plotly_white",
    )
    plotly.offline.plot(
        fig,
        filename=f"{plot_file_path}/{x}_under_{', '.join(group_factor)}_boxplot.html",
    )
