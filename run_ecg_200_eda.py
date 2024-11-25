# %%
from configparser import ConfigParser

import kmedoids
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

# %%

config = ConfigParser()
config.read("projects/ecg_200/ecg_200.ini")
# %%

df = pd.concat(
    objs=(
        pd.DataFrame(
            arff.loadarff(
                "./data/ECG200/ECG200_TRAIN.arff",
            )[0]
        ),
        pd.DataFrame(
            arff.loadarff(
                "./data/ECG200/ECG200_TEST.arff",
            )[0]
        ),
    ),
    axis=0,
).drop(columns=["target"])

k_medoids_train_y = kmedoids.fasterpam(
    diss=euclidean_distances(df),
    medoids=3,
)
k_medoids_test_y = kmedoids.fasterpam(
    diss=euclidean_distances(df),
    medoids=3,
)


fig = go.Figure()

for i in range(df.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=df.columns,
            y=df.iloc[i, :],
            mode="lines",
            line=dict(color="blue", width=1),
            opacity=0.3,
            showlegend=False,
        ),
        row=1,
        col=1,
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
fig = go.Figure()
cnt_red = 0
cnt_blue = 0
cnt_green = 0
for i in range(df.shape[0]):
    if k_medoids_train_y.labels[i] == 0:
        cnt_red += 1
        fig.add_trace(
            go.Scatter(
                x=df.columns,
                y=df.iloc[i, :],
                mode="lines",
                line=dict(color="red", width=1),
                opacity=0.3,
                showlegend=False,
            )
        )
    elif k_medoids_train_y.labels[i] == 1:
        cnt_green += 1
        fig.add_trace(
            go.Scatter(
                x=df.columns,
                y=df.iloc[i, :],
                mode="lines",
                line=dict(color="green", width=1),
                opacity=0.3,
                showlegend=False,
            )
        )
    else:
        cnt_blue += 1
        fig.add_trace(
            go.Scatter(
                x=df.columns,
                y=df.iloc[i, :],
                mode="lines",
                line=dict(color="blue", width=1),
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
pca = PCA(n_components=2).fit(df.T)
tsne_projection = TSNE(
    n_components=2,
    max_iter=500,
    n_iter_without_progress=150,
    n_jobs=2,
    random_state=1122,
).fit_transform(
    df,
    k_medoids_train_y.labels,
)
# %%


k_colors = [
    "red" if result == 0 else "blue" if result == 1 else "green"
    for result in k_medoids_train_y.labels
]

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=False,
    subplot_titles=("PCA (2D with K-medoids)", "T-SNE (2D with K-medoids Labels)"),
)

fig.add_trace(
    go.Scatter(
        x=pca.components_[0],
        y=pca.components_[1],
        mode="markers",
        marker=dict(color=k_colors, size=8, opacity=0.7),
        name="PCA",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=tsne_projection.T[0],
        y=tsne_projection.T[1],
        mode="markers",
        marker=dict(color=k_colors, size=8, opacity=0.7),
        name="T-SNE",
    ),
    row=1,
    col=2,
)

fig.update_layout(
    title="Comparison of PCA and T-SNE (2D with K-medoids Labels)",
    template="plotly_white",
    showlegend=False,
)

fig.update_xaxes(title_text="PCA - Component 1", row=1, col=1)
fig.update_yaxes(title_text="PCA - Component 2", row=1, col=1)
fig.update_xaxes(title_text="T-SNE - Dimension 1", row=1, col=2)
fig.update_yaxes(title_text="T-SNE - Dimension 2", row=1, col=2)

fig.show()
# %%
