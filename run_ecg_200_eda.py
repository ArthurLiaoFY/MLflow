# %%
from configparser import ConfigParser

import kmedoids
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

# %%

config = ConfigParser()
config.read("projects/ecg_200/ecg_200.ini")
# %%

train_df = pd.DataFrame(
    arff.loadarff(
        "./data/ECG200/ECG200_TRAIN.arff",
    )[0]
)
test_df = pd.DataFrame(
    arff.loadarff(
        "./data/ECG200/ECG200_TEST.arff",
    )[0]
)
train_x = train_df.drop(columns=["target"])
test_x = test_df.drop(columns=["target"])
train_y = train_df["target"].astype(int)
test_y = test_df["target"].astype(int)

k_medoids_train_y = kmedoids.fasterpam(
    diss=euclidean_distances(train_x),
    medoids=3,
)
k_medoids_test_y = kmedoids.fasterpam(
    diss=euclidean_distances(test_x),
    medoids=3,
)

# %%
print(train_df.shape)
# %%

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
cnt_red = 0
cnt_blue = 0

for i in range(train_df.shape[0]):

    if train_y[i] == -1:
        cnt_red += 1
        ax1.plot(
            train_x.iloc[i, :],
            color="red",
            alpha=0.3,
        )

    else:
        cnt_blue += 1
        ax2.plot(
            train_x.iloc[i, :],
            color="blue",
            alpha=0.3,
        )

ax1.set_xticks([])
ax2.set_xticks([])


ax1.set_title(f"Label -1 n={cnt_red}")
ax2.set_title(f"Label 1 n={cnt_blue}")

plt.show()
# %%


pca = PCA(n_components=2).fit(train_x.T)
# %%
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
colors = ["red" if result == -1 else "blue" for result in train_y]

# %%
pca = PCA(n_components=2).fit(train_x.T)

fig, (ax1, ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6),
    sharey=False,
)
ax1.scatter(
    *pca.components_,
    c=colors,
)


tsne_projection = TSNE(
    n_components=2,
    max_iter=500,
    n_iter_without_progress=150,
    n_jobs=2,
    random_state=1122,
).fit_transform(
    train_x,
    train_y,
)
ax2.scatter(
    *tsne_projection.T,
    c=colors,
)

ax1.set_title(f"PCA (2D)")
ax2.set_title(f"T-SNE (2D)")
plt.show()

# %%
fig = plt.figure(figsize=(12, 6))
cnt_red = 0
cnt_blue = 0
cnt_green = 0

for i in range(train_df.shape[0]):

    if k_medoids_train_y.labels[i] == 0:
        cnt_red += 1
        plt.plot(
            train_x.iloc[i, :],
            color="red",
            alpha=0.3,
        )
    elif k_medoids_train_y.labels[i] == 1:
        cnt_red += 1
        plt.plot(
            train_x.iloc[i, :],
            color="green",
            alpha=0.3,
        )
    else:
        cnt_blue += 1
        plt.plot(
            train_x.iloc[i, :],
            color="blue",
            alpha=0.3,
        )

plt.xticks([])
plt.show()
# %%

k_colors = [
    "red" if result == 0 else "blue" if result == 1 else "green"
    for result in k_medoids_train_y.labels
]
fig, (k_ax1, k_ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6),
    sharey=False,
)
k_ax1.scatter(
    *pca.components_,
    c=k_colors,
)
k_ax2.scatter(
    *tsne_projection.T,
    c=k_colors,
)

ax1.set_title(f"PCA (2D with k medoids)")
ax2.set_title(f"T-SNE (2D with k medoids)")
plt.show()

# %%
