import numpy as np


def smote(
    curve_array: np.ndarray, label_array: np.ndarray, seed: int = 1122
) -> tuple[np.ndarray, np.ndarray]:
    rand_seed = np.random.RandomState(seed)

    unique_labels, counts = np.unique(label_array, return_counts=True)
    print(unique_labels, counts)

    min_label = unique_labels[np.argmin(counts)]
    min_class_indices = np.where(label_array == min_label)[0]
    print(min_label)

    n_samples_to_add = np.max(counts) - np.min(counts)

    gen_curve_array = np.zeros(
        shape=(n_samples_to_add, curve_array.shape[1], curve_array.shape[2])
    )
    print(gen_curve_array.shape)
    gen_label_array = np.full(n_samples_to_add, min_label)

    for i in range(n_samples_to_add):
        idx_1, idx_2 = rand_seed.choice(min_class_indices, size=2, replace=False)
        alpha = rand_seed.uniform(low=0.0, high=1.0, size=1).item()

        gen_curve_array[i, :, :] = curve_array[idx_1, :, :] + alpha * (
            curve_array[idx_2, :, :] - curve_array[idx_1, :, :]
        )

    return (
        np.concatenate([curve_array, gen_curve_array], axis=0),
        np.concatenate([label_array, gen_label_array], axis=0),
    )


def up_sampling(
    curve_array: np.ndarray, label_array: np.ndarray, seed: int = 1122
) -> tuple[np.ndarray, np.ndarray]:
    rand_seed = np.random.RandomState(seed)

    unique_labels, counts = np.unique(label_array, return_counts=True)

    min_label = unique_labels[np.argmin(counts)]
    min_class_indices = np.where(label_array == min_label)[0]

    n_samples_to_add = np.max(counts) - np.min(counts)

    dup_indices = rand_seed.choice(
        min_class_indices, size=n_samples_to_add, replace=True
    )

    return (
        np.concatenate([curve_array, curve_array[dup_indices]], axis=0),
        np.concatenate([label_array, label_array[dup_indices]], axis=0),
    )
