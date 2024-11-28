import numpy as np


def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0, num_points: int = 10000):
    x_span = np.linspace(mean - 6 * std, x, num_points)
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_span - mean) / std) ** 2)

    return np.trapz(pdf, x_span)
