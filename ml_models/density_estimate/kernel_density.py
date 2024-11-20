from typing import Callable

import numpy as np


def bounded_kernel_wrapper(kernel_func: Callable):
    def wrapper(u: float, h: float, *args):
        if abs(u / h) <= 1:
            return kernel_func(u / h, *args) / h
        else:
            return 0

    return wrapper


@bounded_kernel_wrapper
def uniform(u: float, *args):
    return 0.5


@bounded_kernel_wrapper
def triangular(u: float, *args):
    return 1 - abs(u)


@bounded_kernel_wrapper
def epanechnikov(u: float, *args):
    return 3 * (1 - u**2) / 4


@bounded_kernel_wrapper
def quartic(u: float, *args):
    return 15 * (1 - u**2) ** 2 / 16


@bounded_kernel_wrapper
def triweight(u: float, *args):
    return 35 * (1 - u**2) ** 3 / 32


@bounded_kernel_wrapper
def tricube(u: float, *args):
    return 70 * (1 - abs(u) ** 3) ** 3 / 81


@bounded_kernel_wrapper
def cosine(u: float, *args):
    return np.pi * np.cos(np.pi * u / 2) / 4


def gaussian(u: float, h: float, *args):
    return np.exp(-0.5 * (u / h) ** 2) / np.sqrt(2 * np.pi) / h


def logistic(u: float, h: float, *args):
    return 1 / (np.exp(u / h) + 2 + np.exp(-u / h)) / h


def sigmoid(u: float, h: float, *args):
    return 2 / np.pi / (np.exp(u / h) + np.exp(-u / h)) / h


def silverman(u: float, h: float, *args):
    return (
        np.exp(-abs(u / h) / np.sqrt(2))
        * np.sin(abs(u / h) / np.sqrt(2) + np.pi / 4)
        / 2
        / h
    )
