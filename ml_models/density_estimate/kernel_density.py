import numpy as np


def bounded_kernel_wrapper(kernel_func):
    def wrapper(u, h):
        if abs(u / h) <= 1:
            return kernel_func(u / h) / h
        else:
            return 0

    return wrapper


@bounded_kernel_wrapper
def uniform(u):
    return 0.5


@bounded_kernel_wrapper
def triangular(u):
    return 1 - abs(u)


@bounded_kernel_wrapper
def epanechnikov(u):
    return 3 * (1 - u**2) / 4


@bounded_kernel_wrapper
def quartic(u):
    return 15 * (1 - u**2) ** 2 / 16


@bounded_kernel_wrapper
def triweight(u):
    return 35 * (1 - u**2) ** 3 / 32


@bounded_kernel_wrapper
def tricube(u):
    return 70 * (1 - abs(u) ** 3) ** 3 / 81


@bounded_kernel_wrapper
def cosine(u):
    return np.pi * np.cos(np.pi * u / 2) / 4


def gaussian(u, h):
    return np.exp(-0.5 * (u / h) ** 2) / np.sqrt(2 * np.pi) / h


def logistic(u, h):
    return 1 / (np.exp(u / h) + 2 + np.exp(-u / h)) / h


def sigmoid(u, h):
    return 2 / np.pi / (np.exp(u / h) + np.exp(-u / h)) / h


def silverman(u, h):
    return (
        np.exp(-abs(u / h) / np.sqrt(2))
        * np.sin(abs(u / h) / np.sqrt(2) + np.pi / 4)
        / 2
        / h
    )
