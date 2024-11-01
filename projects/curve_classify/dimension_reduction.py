import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dimension_reduction(
    curve: pd.DataFrame, label: pd.Series, cluster_type: str = "PCA"
):
    match cluster_type:
        case "PCA":
            pca = PCA(n_components=2).fit(curve.T)
            return pca.components_
        case "TSNE":
            tsne_projection = TSNE(
                n_components=2,
                max_iter=500,
                n_iter_without_progress=150,
                n_jobs=2,
                random_state=1122,
            ).fit_transform(
                curve,
                label,
            )
            return tsne_projection.T
        case _:
            raise NotImplementedError
