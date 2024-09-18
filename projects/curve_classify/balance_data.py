import numpy as np


def functional_smote(
    curve_array: np.ndarray,
    label_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    pass


def up_sampling(
    curve_array: np.ndarray,
    label_array: np.ndarray,
    seed: int = 1122,
) -> tuple[np.ndarray, np.ndarray]:
    unique_labels, counts = np.unique(label_array, return_counts=True)

    min_label = unique_labels[np.argmin(counts)]
    min_class_indices = np.where(label_array == min_label)[0]

    n_samples_to_add = np.max(counts) - np.min(counts)

    rng = np.random.default_rng(seed)
    dup_indices = rng.choice(min_class_indices, size=n_samples_to_add, replace=True)

    return (
        np.concatenate([curve_array, curve_array[dup_indices]], axis=0),
        np.concatenate([label_array, label_array[dup_indices]], axis=0),
    )
