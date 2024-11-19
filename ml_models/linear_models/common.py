from abc import ABC, abstractmethod

import numpy as np
import plotly
import plotly.graph_objects as go


class StatisticalModel:
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def plot_residual(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        (self.predict(X=X) - y)
        pass
