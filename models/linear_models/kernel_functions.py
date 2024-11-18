import numpy as np


def bounded_kernel_wrapper(u, h, kernel_func):
    if abs(u / h) <= 1:
        return kernel_func(u / h) / h
    else:
        return 0


@bounded_kernel_wrapper
def uniform_kernel(u):
    return 0.5


@bounded_kernel_wrapper
def triangular_kernel(u):
    return 1 - abs(u)


@bounded_kernel_wrapper
def epanechnikov_kernel(u):
    return 3 * (1 - u**2) / 4


@bounded_kernel_wrapper
def quartic_kernel(u):
    return 15 * (1 - u**2) ** 2 / 16


@bounded_kernel_wrapper
def triweight_kernel(u):
    return 35 * (1 - u**2) ** 3 / 32


@bounded_kernel_wrapper
def tricube_kernel(u):
    return 70 * (1 - abs(u) ** 3) ** 3 / 81


@bounded_kernel_wrapper
def cosine_kernel(u):
    return np.pi * np.cos(np.pi * u / 2) / 4


def gaussian_kernel(u, h):
    return np.exp(-0.5 * (u / h) ** 2) / np.sqrt(2 * np.pi) / h


def logistic_kernel(u, h):
    return 1 / (np.exp(u / h) + 2 + np.exp(-u / h)) / h


def sigmoid_kernel(u, h):
    return 2 / np.pi / (np.exp(u / h) + np.exp(-u / h)) / h


def silverman_kernel(u, h):
    return (
        np.exp(-abs(u / h) / np.sqrt(2))
        * np.sin(abs(u / h) / np.sqrt(2) + np.pi / 4)
        / 2
        / h
    )
