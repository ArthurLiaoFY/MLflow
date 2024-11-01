import numpy as np
import torch
from aeon.datasets import load_from_tsfile
from torch.fft import fft


def load_train_data(
    data_file_path: str,
    augmentation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x_t, train_y = load_from_tsfile(f"{data_file_path}/FaultDetectionA_TRAIN.ts")

    train_x_t = torch.from_numpy(train_x_t)
    train_x_f = fft(train_x_t).abs()
    train_y = torch.from_numpy(train_y.astype(int)).long()

    if augmentation:
        train_x_t = DataTransform_TD(train_x_t)
        train_x_f = DataTransform_FD(train_x_f)  # [7360, 1, 90]

    return (
        train_x_t,
        train_x_f,
        train_y,
    )


def load_test_data(
    data_file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_x_t, test_y = load_from_tsfile(f"{data_file_path}/FaultDetectionA_TEST.ts")

    test_x_t = torch.from_numpy(test_x_t)
    test_y = torch.from_numpy(test_y.astype(int)).long()

    test_x_f = fft(test_x_t).abs()

    return (
        test_x_t,
        test_x_f,
        test_y,
    )


def DataTransform_FD(sample):
    """Weak and strong augmentations in Frequency domain"""
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F


def remove_frequency(x, pertub_ratio=0.0):
    mask = (
        torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio
    )  # maskout_ratio are False
    mask = mask.to(x.device)
    return x * mask


def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (
        1 - pertub_ratio
    )  # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
    pertub_matrix = mask * random_am
    return x + pertub_matrix


def DataTransform_TD(sample, jitter_ratio: float = 0.1):
    """Simplely use the jittering augmentation. Feel free to add more autmentations you want,
    but we noticed that in TF-C framework, the augmentation has litter impact on the final tranfering performance.
    """
    aug = jitter(sample, jitter_ratio)
    return aug


def jitter(x: torch.Tensor, sigma: float = 0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + torch.randn(size=x.shape) * sigma
