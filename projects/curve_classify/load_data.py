import numpy as np
from aeon.datasets import load_from_tsfile


def load_data(
    data_file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x, train_y = load_from_tsfile(f"{data_file_path}/FaultDetectionA_TRAIN.ts")
    test_x, test_y = load_from_tsfile(f"{data_file_path}/FaultDetectionA_TEST.ts")

    return (
        train_x,
        train_y.astype(int),
        test_x,
        test_y.astype(int),
    )
