import numpy as np


def smote(
    curve_array: np.ndarray, label_array: np.ndarray, seed: int = 1122
) -> tuple[np.ndarray, np.ndarray]:
    rand_seed = np.random.RandomState(seed)

    unique_labels, counts = np.unique(label_array, return_counts=True)
    min_label = unique_labels[np.argmin(counts)]
    min_class_indices = np.where(label_array == min_label)[0]

    num_to_generate = np.max(counts) - np.min(counts)

    gen_curve_array = np.zeros((num_to_generate, curve_array.shape[1]))
    gen_label_array = np.full(num_to_generate, min_label)

    for i in range(num_to_generate):
        idx_1, idx_2 = rand_seed.choice(min_class_indices, size=2, replace=False)
        alpha = rand_seed.uniform(low=0.0, high=1.0, size=1)

        gen_curve_array[i] = curve_array[idx_1] + alpha * (
            curve_array[idx_2] - curve_array[idx_1]
        )

    return (
        np.vstack([curve_array, gen_curve_array]),
        np.hstack([label_array, gen_label_array]),
    )


def up_sampling(
    curve_array: np.ndarray, label_array: np.ndarray, seed: int = 1122
) -> tuple[np.ndarray, np.ndarray]:
    unique_labels, counts = np.unique(label_array, return_counts=True)

    min_label = unique_labels[np.argmin(counts)]
    min_class_indices = np.where(label_array == min_label)[0]

    n_samples_to_add = np.max(counts) - np.min(counts)

    dup_indices = np.random.RandomState(seed).choice(
        min_class_indices, size=n_samples_to_add, replace=True
    )

    return (
        np.concatenate([curve_array, curve_array[dup_indices]], axis=0),
        np.concatenate([label_array, label_array[dup_indices]], axis=0),
    )
