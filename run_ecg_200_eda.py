# %%
from configparser import ConfigParser

import pandas as pd
from scipy.io import arff

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

# %%
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
cnt_red = 0
cnt_blue = 0

for i in range(train_df.shape[0]):

    if train_df.iloc[i, -1] == b"-1":
        cnt_red += 1
        ax1.plot(
            train_df.iloc[i, :-1],
            color="red",
            alpha=0.3,
        )

    else:
        cnt_blue += 1
        ax2.plot(
            train_df.iloc[i, :-1],
            color="blue",
            alpha=0.3,
        )

ax1.set_xticks([])
ax2.set_xticks([])


ax1.set_title(f"Label -1 n={cnt_red}")
ax2.set_title(f"Label 1 n={cnt_blue}")

plt.show()
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(train_df.iloc[:, :-1])
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
# %%
