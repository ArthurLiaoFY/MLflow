# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import kmedoids
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import arff
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

from ml_models.linear_models.distance import MahalanobisDistance

# %%

df = (
    pd.concat(
        objs=(
            pd.DataFrame(
                arff.loadarff(
                    "./data/ECG200_TRAIN.arff",
                )[0]
            ),
            pd.DataFrame(
                arff.loadarff(
                    "./data/ECG200_TEST.arff",
                )[0]
            ),
        ),
        axis=0,
    )
    .drop(columns=["target"])
    .reset_index(drop=True)
)

# %%
fig = go.Figure()

for row_idx in range(df.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=df.columns,
            y=df.iloc[row_idx, :],
            mode="lines",
            line=dict(color="blue", width=1),
            opacity=0.3,
            showlegend=False,
        ),
    )


fig.update_layout(
    title_text=f"ECG Trend Data",
    showlegend=False,
    template="plotly_white",
)

fig.update_xaxes(
    title_text="Features",
    tickangle=45,
    showticklabels=False,
)
fig.update_yaxes(title_text="Values")

fig.show()

# %%
medoids = kmedoids.fasterpam(
    diss=euclidean_distances(df),
    medoids=3,
    random_state=1122,
)
color_map = {
    "0": "red",
    "1": "blue",
    "2": "green",
}
# %%
fig = go.Figure()
for row_idx in range(df.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=df.columns,
            y=df.iloc[row_idx, :],
            mode="lines",
            line=dict(color=color_map.get(str(medoids.labels[row_idx])), width=1),
            opacity=0.3,
            showlegend=False,
        )
    )

fig.update_layout(
    title_text=f"ECG Trend Data with K-medoids",
    showlegend=False,
    template="plotly_white",
)

fig.update_xaxes(
    title_text="Features",
    tickangle=45,
    showticklabels=False,
)
fig.update_yaxes(title_text="Values")
fig.show()
# %%
# pca = PCA(n_components=2).fit(df.T)
tsne_projection = TSNE(
    n_components=2,
    max_iter=500,
    n_iter_without_progress=150,
    n_jobs=2,
    random_state=1122,
).fit_transform(
    df,
    medoids.labels,
)
# %%
group_mahalanobis_distance = {
    f"{label}": np.max(
        np.array(
            [
                MahalanobisDistance().inner_distance(
                    X=df.iloc[medoids.labels == label, col].to_numpy()[:, np.newaxis]
                )
                for col in range(df.shape[-1])
            ]
        ),
        axis=0,
    )
    for label in medoids.labels
}


# %%


fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=(
        "Outlier Detection group by K-medoids",
        "T-SNE with K-medoids Labels",
    ),
)
for label in np.unique(medoids.labels):
    fig.add_trace(
        go.Scatter(
            x=tsne_projection[medoids.labels == label, 0],
            y=tsne_projection[medoids.labels == label, 1],
            mode="markers",
            marker=dict(
                color=group_mahalanobis_distance.get(str(label)),
                size=8,
                opacity=0.7,
                colorbar=dict(
                    title="Mahalanobis Distance",
                    titleside="right",
                ),
                colorscale="reds",
                showscale=False,
            ),
            name=f"Label: {str(label)}",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


fig.add_trace(
    go.Scatter(
        x=tsne_projection.T[0],
        y=tsne_projection.T[1],
        mode="markers",
        marker=dict(
            color=[color_map.get(str(result)) for result in medoids.labels],
            size=8,
            opacity=0.7,
        ),
        showlegend=False,
    ),
    row=1,
    col=2,
)

fig.update_layout(
    title="T-SNE (2D) with Mahalanobis Distance base Outlier Identification",
    template="plotly_white",
    showlegend=False,
)

fig.update_xaxes(title_text="T-SNE - Dimension 1", row=1, col=1)
fig.update_yaxes(title_text="T-SNE - Dimension 2", row=1, col=1)
fig.update_xaxes(title_text="T-SNE - Dimension 1", row=1, col=2)

fig.show()

# %%

fig = go.Figure()

for label in np.unique(medoids.labels):
    fig.add_trace(
        go.Histogram(
            x=group_mahalanobis_distance.get(str(label)),
            nbinsx=30,
            marker=dict(
                color=color_map.get(str(label)), line=dict(width=1, color="black")
            ),
            opacity=0.7,
            name=f"Group {str(label)}",
        )
    )


fig.update_layout(
    title="Histogram of Mahalanobis Distances (t-SNE)",
    xaxis_title="Mahalanobis Distance",
    yaxis_title="Frequency",
    template="plotly_white",
    showlegend=False,
)

fig.update_layout(barmode="stack")
fig.show()

# %%


# %%
group_mahalanobis_distance_threshold = {
    "0": 10,
    "1": 5,
    "2": 10,
}
outlier_idx = [
    idx
    for label in np.unique(medoids.labels)
    for idx in (
        df.loc[medoids.labels == label, :].loc[
            group_mahalanobis_distance.get(str(label))
            >= group_mahalanobis_distance_threshold.get(str(label)),
            :,
        ]
    ).index.to_list()
]

# %%
fig = make_subplots(
    rows=1,
    cols=3,
    shared_yaxes=True,
    subplot_titles=["Label 0", "Label 1", "Label 2"],
    horizontal_spacing=0.1,
)

for row_idx in range(df.shape[0]):
    width = 4 if row_idx in outlier_idx else 2
    opacity = 1 if row_idx in outlier_idx else 0.3
    to_black = True if row_idx in outlier_idx else False

    fig.add_trace(
        go.Scatter(
            x=list(range(df.shape[1])),
            y=df.loc[row_idx, :],
            mode="lines",
            line=dict(
                color=(
                    color_map.get(str(medoids.labels[row_idx]))
                    if not to_black
                    else "black"
                ),
                width=width,
            ),
            opacity=opacity,
            showlegend=False,
        ),
        row=1,
        col=int(medoids.labels[row_idx] + 1),
    )


fig.update_layout(
    title_text=f"ECG Trend Data with Outliers Highlighted",
    showlegend=False,
    template="plotly_white",
)

fig.update_xaxes(
    title_text="Features",
    tickangle=45,
    showticklabels=False,
)
fig.update_yaxes(title_text="Values", row=1, col=1)

fig.show()
# %%
fig = go.Figure()

for row_idx, cluster_label in zip(
    df.loc[~df.index.isin(outlier_idx), :].index,
    medoids.labels[
        [idx for idx in range(len(medoids.labels)) if idx not in outlier_idx]
    ],
):
    fig.add_trace(
        go.Scatter(
            x=list(range(df.shape[1])),
            y=df.loc[row_idx, :],
            mode="lines",
            line=dict(
                color=color_map.get(str(cluster_label)),
                width=1,
            ),
            opacity=0.3,
            showlegend=False,
        )
    )


fig.update_layout(
    title_text=f"ECG Trend Data with Outliers filtered",
    showlegend=False,
    template="plotly_white",
)

fig.update_xaxes(
    title_text="Features",
    tickangle=45,
    showticklabels=False,
)
fig.update_yaxes(title_text="Values")

fig.show()

# %%
